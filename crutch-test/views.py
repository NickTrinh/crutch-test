"""Score more simulated students on a run's existing dialogs; no new dialogs.

Each view changes only what the student carries into its unaided test on the
new variant:

  M  tutor turns hidden, student turns kept (EduardoRL's test), rescored here
     with the same cap and logging as the other views
  C  M minus every student sentence that copies the tutor: 50% or more of its
     word 6-grams appear in some tutor turn. Circular by construction (it
     removes what the leak judge looks for), so a reference, not a candidate
  K  only a note of at most 120 tokens, which the student writes right after
     the dialog with the whole dialog in view

Usage: python views.py <EduardoRL checkout> runs/pilot50 [--views M,C,K] [--arms N,H,T,S,W]
Writes runs/<run>/views/<arm>_<view>.json and logs every answer to
runs/<run>/answers.jsonl. Finished (arm, view) pairs are skipped unless --force,
and so are arms the run has no dialogs for.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NOTE_PROMPT = (
    "The conversation with the teacher has ended. Write a short note to your future self, "
    "at most 120 tokens, with what you need to solve problems like this one on your own. "
    "You will not see this conversation again, only this note."
)
NOTE_TOKENS = 120
COPY_N, COPY_SHARE = 6, 0.5
SAMPLES = {"M": 16, "C": 8, "K": 16}  # proposal: 16 for M and K, 8 for C
HIDDEN = "(hidden)"
REMOVED = "(removed)"

_TOKEN = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def ngrams(text: str, n: int = COPY_N) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def copy_filter(text: str, tutor_grams: set) -> tuple[str, int, int]:
    """Drop sentences that copy the tutor. Returns (kept text, dropped, total)."""
    kept, dropped = [], 0
    for sentence in (s.strip() for s in _SENTENCE.split(text)):
        if not sentence:
            continue
        grams = ngrams(sentence)
        if grams and len(grams & tutor_grams) / len(grams) >= COPY_SHARE:
            dropped += 1
        else:
            kept.append(sentence)
    return (" ".join(kept) or REMOVED), dropped, dropped + len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eduardo")
    ap.add_argument("run")
    ap.add_argument("--views", default="M,C,K")
    ap.add_argument("--arms", default="N,H,T,S,W")
    ap.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-retries", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eduardo).resolve()))
    sys.path.insert(0, str(HERE))
    import openrouter_routes
    import posttest
    from eduardo import utils as eduardo_utils
    from eduardo.metrics import correctness
    from eduardo.utils import call_openai_with_retries, get_async_openai_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("crutch.views")

    run = Path(args.run)
    out_dir = run / "views"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CRUTCH_ANSWERS", str(run / "answers.jsonl"))
    student_prompt = correctness._STUDENT_SYSTEM_PROMPT

    def tutor_text(content: str) -> str:
        return eduardo_utils.strip_teacher_controls(content, assume_thinking=False)

    def student_view(rec: dict, final_prompt: str, *, tutor: str, student=None) -> list[dict]:
        """EduardoRL's final-attempt layout: the student speaks as assistant."""
        msgs = [{"role": "system", "content": student_prompt.format(problem=rec["problem"])}]
        for m in rec["raw_messages"]:
            if m["role"] == "teacher":
                msgs.append({"role": "user", "content": HIDDEN if tutor == "hidden" else tutor_text(m["content"])})
            else:
                msgs.append({"role": "assistant", "content": m["content"] if student is None else student(m["content"])})
        msgs.append({"role": "user", "content": final_prompt})
        return msgs

    async def note_once(client, model, msgs, temperature, max_tokens):
        resp = await client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""

    async def score_dialog(rec, arm, view, client, sem, retry):
        key = openrouter_routes.problem_key(rec["problem"])
        final = correctness._render_final_prompt(rec["transfer_problem"])
        extra = {}
        if view == "M":
            msgs = student_view(rec, final, tutor="hidden")
        elif view == "C":
            grams = set().union(*(ngrams(tutor_text(m["content"]))
                                  for m in rec["raw_messages"] if m["role"] == "teacher"))
            counts = [0, 0]

            def filtered(text):
                kept, dropped, total = copy_filter(text, grams)
                counts[0] += dropped
                counts[1] += total
                return kept

            msgs = student_view(rec, final, tutor="hidden", student=filtered)
            extra = {"removed": counts[0], "sentences": counts[1]}
        else:  # K
            note = await call_openai_with_retries(
                note_once, client, openrouter_routes.STUDENT,
                student_view(rec, NOTE_PROMPT, tutor="visible"), 0.6, NOTE_TOKENS,
                label="note", use_fallback=True, fallback=None, semaphore=sem, **retry)
            if note is None:
                return {"problem": key, "score": None, "unboxed": 0, "failed": SAMPLES["K"], "note": None}
            msgs = [{"role": "system", "content": student_prompt.format(problem=rec["problem"])},
                    {"role": "assistant", "content": note},
                    {"role": "user", "content": final}]
            extra = {"note": note}
        score, unboxed, failed = await posttest.sample_and_grade(
            client, openrouter_routes.STUDENT, msgs, str(rec["transfer_answer"]), SAMPLES[view],
            semaphore=sem, retry_kwargs=retry,
            log={"tag": f"{arm}_{view}", "problem": key, "test": "variant", "view": view})
        return {"problem": key, "score": score, "unboxed": unboxed, "failed": failed, **extra}

    async def run_all():
        client = get_async_openai_client(args.api_base, args.api_key_env)
        sem = asyncio.Semaphore(args.concurrency)
        retry = {"num_retries": args.num_retries}

        async def job(arm, view):
            path = out_dir / f"{arm}_{view}.json"
            if path.exists() and not args.force:
                log.info("%s_%s: done already", arm, view)
                return
            src = sorted(glob.glob(str(run / "eval" / f"transfer_conversations_{arm}_*.jsonl")))[-1]
            recs = [json.loads(line) for line in open(src, encoding="utf-8")]
            dialogs = await asyncio.gather(*(score_dialog(r, arm, view, client, sem, retry) for r in recs))
            path.write_text(json.dumps({"arm": arm, "view": view, "samples": SAMPLES[view],
                                        "source": Path(src).name, "dialogs": dialogs}, indent=1))
            scored = [d["score"] for d in dialogs if d["score"] is not None]
            log.info("%s_%s: %d dialogs, mean %.3f, %d failed calls", arm, view, len(dialogs),
                     sum(scored) / max(1, len(scored)), sum(d["failed"] for d in dialogs))

        # S and W are optional arms: skip any the run has no dialogs for.
        arms = [a for a in args.arms.split(",")
                if glob.glob(str(run / "eval" / f"transfer_conversations_{a}_*.jsonl"))]
        log.info("arms with dialogs: %s", ",".join(arms))
        await asyncio.gather(*(job(a, v) for a in arms for v in args.views.split(",")))

    asyncio.run(run_all())
    if os.environ.get("CRUTCH_LEDGER"):
        openrouter_routes.dump_ledger(os.environ["CRUTCH_LEDGER"], "views")


if __name__ == "__main__":
    main()
