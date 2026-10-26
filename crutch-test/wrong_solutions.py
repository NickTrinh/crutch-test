"""Write the positive control's confidently wrong solutions (arm W).

For each problem the tutor model writes a worked solution in the GPT Base
tutor's style, with one plausible mistake and a confident final \\boxed{}
answer. A solution is kept only if its final answer parses and is graded
wrong against the reference, and its text does not hint at the mistake. Each
problem gets up to --tries attempts; a problem with no usable solution is left
out of W only, and the report counts it.

Usage: python wrong_solutions.py <EduardoRL checkout> --parquet .data/arms/heldout_W.parquet \
           --out runs/<run>/wrong_solutions.json --w-parquet runs/<run>/W.parquet
The JSON keeps every rejected try's reason; W's parquet has the rows, among
the first --num-problems, that have a solution. Set CRUTCH_LEDGER to log cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Importing it routes Claude (no temperature, low effort), waits out 429s and keeps the ledger.
import openrouter_routes  # noqa: E402
from openrouter_routes import LIVE_TUTOR, problem_key  # noqa: E402

PROMPT = (
    "You are preparing material for a study of how tutoring affects learning. Write a worked solution "
    "to the math problem below, as a confident tutor would when a student asks \"What is the answer?\". "
    "The solution must contain exactly one plausible mistake, such as a sign error, a misapplied rule or "
    "a missed case, and that mistake must lead to a wrong final answer. Present every step with "
    "confidence: do not mention, hint at or correct the mistake. The correct final answer is {answer}; "
    "yours must be different.\n\n"
    "Write in Github supported Markdown. Wrap inline math with '$' and put block math between '$$' lines. "
    "When you write a numbered list, use 'First,' instead of '1.'. "
    "End with this line: Final answer: $\\boxed{{your final answer}}$\n\n"
    "Problem: {problem}"
)
# A solution that tells the student it is wrong is no longer "confidently" wrong.
DISCLOSURE = re.compile(r"\b(?:error|errors|mistake|mistakes|wrong|incorrect|oops|wait|actually|typo)\b", re.I)


def check(text: str | None, answer: str) -> tuple[str | None, str]:
    """(Why this solution cannot serve W, or None if it can; its final answer).

    Needs EduardoRL on sys.path (main() and the tests add it).
    """
    from eduardo.utils import _timeout, grade_math_answer
    from math_verify import parse, verify

    if not text:
        return "empty reply", ""
    if DISCLOSURE.search(text):
        return "hints at its mistake", ""
    grade = grade_math_answer(text, answer)
    if grade["incorrect_format"]:
        return "no boxed answer", ""
    if grade["acc"]:
        return "final answer is correct", grade["pred"]
    # EduardoRL's grader parses bare strings and falls back to string match,
    # which calls \sqrt3 and \sqrt{3} different. Wrapped in $...$ both sides
    # parse as LaTeX; an answer that cannot be checked is not used.
    try:
        with _timeout(seconds=5):
            gold, guess = parse(f"${answer}$"), parse(f"${grade['pred']}$")
            same = bool(gold and guess and verify(gold, guess))
    except Exception:
        return "answer cannot be checked", grade["pred"]
    if not (gold and guess):
        return "answer cannot be checked", grade["pred"]
    return ("final answer is correct" if same else None), grade["pred"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eduardo", help="EduardoRL checkout")
    ap.add_argument("--parquet", required=True, help="the split's W parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-parquet", required=True, help="where to write W's rows")
    ap.add_argument("--model", default=LIVE_TUTOR)
    ap.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--num-problems", type=int, default=-1)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-retries", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eduardo).resolve()))
    from eduardo.utils import call_openai_with_retries, get_async_openai_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    df = pd.read_parquet(args.parquet)
    if args.num_problems > 0:
        df = df.iloc[: args.num_problems]
    problems = {problem_key(e["problem"]): e for e in df["extra_info"]}

    async def ask(client, sem, e):
        async def _call():
            resp = await client.chat.completions.create(
                model=args.model, max_tokens=4096,
                messages=[{"role": "user", "content": PROMPT.format(answer=e["answer"], problem=e["problem"])}])
            return resp.choices[0].message.content or ""
        return await call_openai_with_retries(_call, label="wrong_solution", use_fallback=True, fallback=None,
                                              semaphore=sem, num_retries=args.num_retries)

    async def generate():
        client = get_async_openai_client(args.api_base, args.api_key_env)
        sem = asyncio.Semaphore(args.concurrency)
        solutions, rejected, failed = {}, {k: [] for k in problems}, 0
        pending = list(problems)
        for _ in range(args.tries):
            texts = await asyncio.gather(*(ask(client, sem, problems[k]) for k in pending))
            for k, text in zip(pending, texts):
                if text is None:
                    failed += 1
                    rejected[k].append("call failed after retries")
                    continue
                why, pred = check(text, str(problems[k]["answer"]))
                if why:
                    rejected[k].append(why)
                else:
                    solutions[k] = {"text": text.strip(), "pred": pred, "answer": str(problems[k]["answer"]),
                                    "tries": len(rejected[k]) + 1}
            pending = [k for k in pending if k not in solutions]
            if not pending:
                break
        return solutions, {k: rejected[k] for k in pending}, failed

    solutions, dropped, failed = asyncio.run(generate())
    if os.environ.get("CRUTCH_LEDGER"):
        openrouter_routes.dump_ledger(os.environ["CRUTCH_LEDGER"], "wrong_solutions")
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "prompt": PROMPT, "tries": args.tries, "problems": len(problems),
         "solutions": solutions, "dropped": dropped}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[wrong] {len(solutions)}/{len(problems)} problems have a checked wrong solution; "
          f"{len(dropped)} left out of W -> {args.out}")
    # No W parquet unless this step is complete, so a rerun of pilot.sh redoes it.
    if failed:
        raise SystemExit(f"[wrong] {failed} calls failed after retries; rerun this step")
    if not solutions:
        raise SystemExit(f"[wrong] no problem got a usable wrong solution; see the reasons in {args.out}")
    keep = [problem_key(e["problem"]) in solutions for e in df["extra_info"]]
    df[keep].reset_index(drop=True).to_parquet(args.w_parquet)


if __name__ == "__main__":
    main()
