"""Grade the learning student's answers and file them as view L of a run.

Reads results/learner_outputs_<run>.json.gz (written by learner_modal.py),
grades every answer with EduardoRL's grader, and writes:
  runs/<run>/views/{N,H,T}_L.json   per dialog: variant score (the view's
                                    score), same-problem and unrelated scores,
                                    trained tokens, loss drop
  runs/<run>/views/base_L.json      the frozen model on the same tests
  runs/<run>/answers.jsonl          every answer, tagged <arm>_L
analyze.py then reports L beside M, C and K, with learning and drift checks.

Usage: python learner_grade.py <EduardoRL checkout> runs/pilot50v2
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eduardo")
    ap.add_argument("run")
    ap.add_argument("--outputs", default=None, help="default: results/learner_outputs_<run>.json.gz")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.eduardo).resolve()))
    from eduardo.utils import grade_math_answer

    run = Path(args.run)
    src = Path(args.outputs) if args.outputs else HERE / "results" / f"learner_outputs_{run.name}.json.gz"
    with gzip.open(src, "rt", encoding="utf-8") as f:
        out = json.load(f)

    scores: dict[tuple, dict] = {}
    with open(run / "answers.jsonl", "a", encoding="utf-8") as log:
        for a in out["answers"]:
            grades = [grade_math_answer(t, a["answer"]) for t in a["texts"]]
            # The frozen model's tests all share adapter "base", so the problem key is part of the index.
            scores[(a["adapter"], a["key"], a["kind"])] = {
                "score": statistics.fmean(int(g["acc"]) for g in grades) if grades else None,
                "unboxed": sum(int(g["incorrect_format"]) for g in grades),
            }
            for i, (t, g) in enumerate(zip(a["texts"], grades)):
                log.write(json.dumps({"tag": f"{a['arm']}_L", "problem": a["key"], "test": a["kind"], "view": "L",
                                      "adapter": a["adapter"], "i": i, "finish": a["finish"][i],
                                      "tokens": a["tokens"][i], "pred": g["pred"], "acc": int(g["acc"]),
                                      "unboxed": int(g["incorrect_format"]), "text": t}) + "\n")

    vdir = run / "views"
    vdir.mkdir(parents=True, exist_ok=True)
    per_arm = defaultdict(list)
    for i, t in enumerate(out["train"]):
        v, s, u = (scores.get((i, t["key"], k), {}) for k in ("variant", "same", "unrelated"))
        per_arm[t["arm"]].append({"problem": t["key"], "score": v.get("score"), "unboxed": v.get("unboxed", 0),
                                  "failed": 0, "same": s.get("score"), "unrelated": u.get("score"),
                                  "n_train_tokens": t["n_train_tokens"], "drop": t["drop"]})
    for arm, dialogs in per_arm.items():
        (vdir / f"{arm}_L.json").write_text(json.dumps(
            {"arm": arm, "view": "L", "samples": 16, "source": src.name, "lr": out["lr"], "dialogs": dialogs}, indent=1))
    base = defaultdict(dict)
    for (adapter, key, kind), s in scores.items():
        if adapter == "base":
            base[key][kind] = s["score"]
    (vdir / "base_L.json").write_text(json.dumps(
        {"lr": out["lr"], "calibration": out["calibration"], "minutes": {k: v for k, v in out.items() if k.endswith("minutes")},
         "base": base}, indent=1))
    print(f"[learner] graded {sum(len(a['texts']) for a in out['answers'])} answers, lr {out['lr']}; "
          f"wrote {', '.join(f'{a}_L' for a in sorted(per_arm))} and base_L to {vdir}")


if __name__ == "__main__":
    main()
