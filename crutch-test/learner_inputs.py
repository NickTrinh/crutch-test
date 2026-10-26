"""Build the learning student's inputs from a finished run's dialogs.

For every dialog: the student's view of it (tutor turns as user, the student's
own turns as assistant, thinking and control tokens stripped), which of those
turns it trains on (its own, minus the scripted opener), and its three tests:
the new variant (16 samples), the same problem restated (8), and an unrelated
problem, another dialog's variant, as the drift check (8). Tests show no
transcript: only the student prompt and the test question.

Usage: python learner_inputs.py <EduardoRL checkout> runs/pilot50v2
Writes results/learner_inputs_<run>.json.gz, which learner_modal.py reads.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = {"name": "RedHatAI/Llama-3.1-8B-Instruct", "revision": "83c927477af31b2c66de4c0210ae5fa04b28289b"}
SAMPLES = {"variant": 16, "same": 8, "unrelated": 8}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("eduardo")
    ap.add_argument("run")
    ap.add_argument("--arms", default="N,H,T")
    ap.add_argument("--out", default=None, help="default: results/learner_inputs_<run>.json.gz")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.eduardo).resolve()))
    sys.path.insert(0, str(HERE))
    import learner_core
    import posttest
    from eduardo import utils as eduardo_utils
    from eduardo.metrics import correctness
    from openrouter_routes import problem_key

    persona = correctness._STUDENT_SYSTEM_PROMPT
    run = Path(args.run)

    def tutor_text(content: str) -> str:
        return eduardo_utils.strip_teacher_controls(content, assume_thinking=False).strip()

    arms = args.arms.split(",")
    recs_by_arm = {}
    for arm in arms:
        src = sorted(glob.glob(str(run / "eval" / f"transfer_conversations_{arm}_*.jsonl")))[-1]
        recs_by_arm[arm] = sorted((json.loads(line) for line in open(src, encoding="utf-8")),
                                  key=lambda r: problem_key(r["problem"]))
    # The unrelated problem: half the sorted problem list away, the same one for every arm.
    by_key = {problem_key(r["problem"]): r for recs in recs_by_arm.values() for r in recs}
    keys = sorted(by_key)
    unrelated = {k: by_key[keys[(i + len(keys) // 2) % len(keys)]] for i, k in enumerate(keys)}

    dialogs = []
    for arm in arms:
        for rec in recs_by_arm[arm]:
            other = unrelated[problem_key(rec["problem"])]
            system = {"role": "system", "content": persona.format(problem=rec["problem"]).strip()}
            messages, trainable = [system], [False]
            for i, m in enumerate(rec["raw_messages"]):
                if m["role"] == "teacher":
                    messages.append({"role": "user", "content": tutor_text(m["content"])})
                    trainable.append(False)
                else:
                    messages.append({"role": "assistant", "content": m["content"].strip()})
                    trainable.append(i > 0)  # the first student turn is the scripted opener
            ask = lambda prompt: [system, {"role": "user", "content": prompt.strip()}]
            dialogs.append({
                "arm": arm,
                "key": problem_key(rec["problem"]),
                "messages": messages,
                "trainable": trainable,
                "tests": [
                    {"kind": "variant", "n": SAMPLES["variant"], "answer": str(rec["transfer_answer"]),
                     "messages": ask(correctness._render_final_prompt(rec["transfer_problem"]))},
                    {"kind": "same", "n": SAMPLES["same"], "answer": str(rec["answer"]),
                     "messages": ask(posttest.SAME_PROBLEM_PROMPT.replace("{problem}", rec["problem"]))},
                    {"kind": "unrelated", "n": SAMPLES["unrelated"], "answer": str(other["transfer_answer"]),
                     "messages": ask(correctness._render_final_prompt(other["transfer_problem"]))},
                ],
            })

    data = {
        "run": run.name,
        "model": MODEL,
        "settings": {"steps": learner_core.STEPS, "loss_scale": learner_core.LOSS_SCALE,
                     "lr_grid": list(learner_core.LR_GRID), "target_drop": learner_core.TARGET_DROP,
                     "lora": learner_core.LORA, "calibrate_on": ["N", "H"],
                     "calibration_dialogs_per_arm": learner_core.CALIBRATION_DIALOGS,
                     "temperature": 0.6, "max_tokens": posttest.MAX_TOKENS},
        "dialogs": dialogs,
    }
    out = Path(args.out) if args.out else HERE / "results" / f"learner_inputs_{run.name}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    trainable_turns = sum(sum(d["trainable"]) for d in dialogs)
    print(f"[learner] {len(dialogs)} dialogs, {trainable_turns} trainable student turns -> {out}")


if __name__ == "__main__":
    main()
