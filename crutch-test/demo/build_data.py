"""Build the dialog explorer's data.json from a graded pilot run.

Usage, from the repo root after the README's demo steps (setup.sh, then
learner_grade.py and analyze.py on a fresh unpack of the released logs):
    crutch-test/.venv/bin/python crutch-test/demo/build_data.py crutch-test/.deps/EduardoRL crutch-test/runs/pilot50v2

Writes crutch-test/demo/data.json: per problem, the three dialogs, what each
simulator carried into the test, the per-view scores, and one sampled answer
per score. It fails unless the 50-problem averages it implies equal the ones
in results/pilot50v2.json, so the page cannot show numbers the report does not.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from openrouter_routes import problem_key  # noqa: E402
from views import COPY_SHARE, _SENTENCE, ngrams  # noqa: E402  (the C view's copy rule, reused as is)

ARMS = ("N", "H", "T")
# The T dialog was judged a handover, the frozen M student gains on the variant
# after answers and the learning student does not, and the T dialog has two turns.
FEATURED = "322b6eb257bb"
# Page column -> (answers.jsonl tag suffix, test). P is practice: the same problem
# with the dialog in view; the others are the new variant, taken without help.
VIEWS = {"P": ("open", "same"), "M": ("masked", "variant"), "C": ("C", "variant"),
         "K": ("K", "variant"), "L": ("L", "variant")}
TAIL = 700  # characters kept from each sample answer; the end holds the boxed answer


def one(pattern: str) -> Path:
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {len(hits)}")
    return Path(hits[0])


def load_run(run: Path) -> tuple[dict, dict, dict, dict]:
    """Dialogs, handover verdicts, mean scores and first samples, keyed by problem."""
    dialogs, leaked = defaultdict(dict), defaultdict(dict)
    for arm in ARMS:
        for line in one(str(run / "eval" / f"transfer_conversations_{arm}_*.jsonl")).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                dialogs[arm][problem_key(rec["problem"])] = rec
        ev = json.loads(one(str(run / "eval" / f"transfer_eval_{arm}_masked_*.json")).read_text(encoding="utf-8"))
        for rec in ev["conversations"]:
            leaked[arm][problem_key(rec["problem"])] = int(rec["leaked"])
    accs, first = defaultdict(list), {}
    for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cell = (r["tag"], r["problem"], r["test"])
        accs[cell].append(r["acc"])
        if r["i"] == 0:
            first[cell] = r
    means = {cell: sum(v) / len(v) for cell, v in accs.items()}
    return dialogs, leaked, means, first


def marked_sentences(rec: dict, strip) -> tuple[list[list[list]], int]:
    """Each student turn split into [sentence, copied] pairs by the C view's rule."""
    grams = set().union(*(ngrams(strip(m["content"], assume_thinking=False))
                          for m in rec["raw_messages"] if m["role"] == "teacher"))
    turns, removed = [], 0
    for m in rec["raw_messages"]:
        if m["role"] != "student":
            continue
        marks = []
        for sentence in (s.strip() for s in _SENTENCE.split(m["content"])):
            if sentence:
                g = ngrams(sentence)
                copied = bool(g) and len(g & grams) / len(g) >= COPY_SHARE
                removed += copied
                marks.append([sentence, copied])
        turns.append(marks)
    return turns, removed


def check_averages(problems: list[dict], report: dict) -> None:
    """The page's per-problem scores must average to the report's effects."""
    for col in VIEWS:
        for arm in ("T", "H"):
            ours = sum(p["scores"][arm][col] - p["scores"]["N"][col] for p in problems) / len(problems)
            contrast = f"{arm}-N"
            theirs = (report["contrasts"][contrast]["P"] if col == "P" else report["views"]["views"][col][contrast])["mean"]
            if abs(ours - theirs) > 1e-9:
                raise SystemExit(f"{col} {contrast}: page average {ours:.6f} differs from the report's {theirs:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("eduardo", type=Path, help="EduardoRL checkout (setup.sh puts it in crutch-test/.deps)")
    ap.add_argument("run", type=Path, help="a graded run folder, e.g. crutch-test/runs/pilot50v2")
    args = ap.parse_args()
    sys.path.insert(0, str(args.eduardo.resolve()))
    from eduardo.utils import strip_teacher_controls

    dialogs, leaked, means, first = load_run(args.run)
    notes, c_counts = {}, {}
    for arm in ARMS:
        for d in json.loads((args.run / "views" / f"{arm}_K.json").read_text(encoding="utf-8"))["dialogs"]:
            notes[arm, d["problem"]] = d["note"]
        for d in json.loads((args.run / "views" / f"{arm}_C.json").read_text(encoding="utf-8"))["dialogs"]:
            c_counts[arm, d["problem"]] = (d["removed"], d["sentences"])
    learner = json.loads(gzip.decompress((ROOT / "results" / "learner_outputs_pilot50v2.json.gz").read_bytes()))
    trained = {(t["arm"], t["key"]): t for t in learner["train"]}
    report = json.loads((ROOT / "results" / "pilot50v2.json").read_text(encoding="utf-8"))

    problems = []
    for key, rec in dialogs["T"].items():
        p = {"key": key, "problem": rec["problem"], "answer": rec["answer"],
             "variant": rec["transfer_problem"], "variant_answer": rec["transfer_answer"],
             "arms": {}, "scores": {}, "samples": {}}
        for arm in ARMS:
            d = dialogs[arm][key]
            turns, removed = marked_sentences(d, strip_teacher_controls)
            if (removed, sum(len(t) for t in turns)) != c_counts[arm, key]:
                raise SystemExit(f"{arm} {key}: copy rule gives {removed} removed, the C view logged {c_counts[arm, key]}")
            t = trained[arm, key]
            p["arms"][arm] = {
                "turns": [{"who": "tutor" if m["role"] == "teacher" else "student", "text": m["content"]}
                          for m in d["messages"]],
                "handover": leaked[arm][key], "end": d["end_reason"],
                "student_sentences": turns, "note": notes[arm, key],
                "trained_tokens": t["n_train_tokens"], "loss_drop": round(t["drop"], 4),
            }
            p["scores"][arm], p["samples"][arm] = {}, {}
            for col, (suffix, test) in VIEWS.items():
                cell = (f"{arm}_{suffix}", key, test)
                p["scores"][arm][col] = means[cell]
                s = first[cell]
                text = s["text"]
                p["samples"][arm][col] = {"text": text if len(text) <= TAIL else "…" + text[-TAIL:],
                                          "pred": s["pred"], "correct": bool(s["acc"])}
        problems.append(p)
    check_averages(problems, report)

    summary = {"P": {c: report["contrasts"][c]["P"] for c in ("T-N", "H-N")}}
    summary.update({v: {c: report["views"]["views"][v][c] for c in ("T-N", "H-N")} for v in ("M", "C", "K", "L")})
    data = {"featured": FEATURED, "learning_rate": learner["lr"], "summary": summary,
            "problems": sorted(problems, key=lambda p: p["key"] != FEATURED)}
    out = HERE / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(problems)} problems); averages match the report")


if __name__ == "__main__":
    main()
