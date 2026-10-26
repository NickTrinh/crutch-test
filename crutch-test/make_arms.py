"""Draw the pilot's problems and write one parquet per tutor arm.

Reads EduardoRL's processed test.parquet, keeps rows whose transfer variant has
a cold solve rate in [1/16, 0.75], and draws a seeded 50-problem dev set
(PILOT-50) and a 300-problem held-out set. Every arm gets the same problems and
the same student opener, "What is the answer?" (the most common first message
in the human trial's GPT Base arm), so only the tutor differs:

  T  GPT Base prompt from Bastani et al. (PNAS 2025): follows the student's lead.
  H  GPT Tutor prompt from the same trial: hints only, reference answer as notes.
  N  No-AI control: crutch/null-tutor only says "Keep going. Show your next step."
  S  Try first: crutch/try-first asks for an attempt, then replays T's first
     turn; T's prompt for the live tutor after that.
  W  Positive control: crutch/wrong-solution gives a confidently wrong solution,
     then N's filler.

Usage: python make_arms.py --test-parquet .data/test.parquet --out-dir .data/arms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from openrouter_routes import (BASE_PROBLEM_ANCHOR, END_RULE_ANCHOR, NULL_SYSTEM_PREFIX,
                               WRONG_SYSTEM_PREFIX, problem_key)

HERE = Path(__file__).resolve().parent

# The two tutor prompts are the human trial's (github.com/obastani/GenAICanHarmLearning).
# That repository has no license, so they are not copied here. setup.sh downloads
# its conversation log (text_analysis/data/raw/valid_student_data.csv at commit
# 2f63dae, 40 MB), and tutor_prompts() cuts each prompt out of one system turn:
# treatment "vanilla" is GPT Base, "aug" is GPT Tutor. The CSV lost its backslash
# escapes ("\n" became "n"), so those newlines are restored; typos are kept.
# Big-Math has no teacher notes, so GPT Tutor's notes are the answer alone.
BASTANI_CSV = HERE / ".deps" / "GenAICanHarmLearning" / "valid_student_data.csv"
# name: (CSV row, parts, SHA-256 of the template every pilot ran with). A part is
# (start, end, offsets of restored newlines) in that row's message, or text of ours.
PROMPT_RECIPES = {
    "GPT_BASE": (0, [(1, 435, (332, 335)), "{problem}"],
                 "a82a2db2bd341300a4e7fa934e5045fbc47d1939126d6c27abdece28b201fea9"),
    "GPT_TUTOR": (171, [(1, 340, (120, 123, 185)), "{topic}",
                        (377, 2093, (177, 276, 838, 1025, 1167, 1358, 1526, 1657, 1658)), "{kind}",
                        (2112, 2123, ()), "{problem}", (2264, 2390, (49, 50, 99)), "{answer}", "."],
                  "4a3333918731f7fcb31dbc4d84de17b6cdbb14e5a03bab56170e3ae3492820fc"),
}
# Big-Math rows carry no lesson topic, so the topic slot stays generic.
TOPIC = "solving problems like the one below"
# Human students left when they were done; the simulated student cannot, and
# it asks for practice because its prompt says a test is coming. Without these
# rules the tutor fills the remaining turns with new practice problems (18 of
# 50 T dialogs in PILOT-50 v1), which is near-transfer practice, not telling.
# The same sentences go on both tutor prompts.
EOC_RULE = ("\n\nDiscuss only this problem: do not give the student new or practice problems. "
            "When the student has no further questions about this problem, "
            "end your reply with <end_of_conversation>.")
OPENER = "What is the answer?"
# S's in-process tutor finds the problem between BASE_PROBLEM_ANCHOR and this
# (openrouter_routes.problem_of); tutor_prompts() checks the other end.
assert EOC_RULE.startswith(END_RULE_ANCHOR)

ARMS = ("T", "H", "N", "S", "W")
LO, HI = 1 / 16, 0.75

_prompts: dict[str, str] = {}


def tutor_prompts() -> dict[str, str]:
    """GPT Base and GPT Tutor as format templates, rebuilt from the trial's log on first use."""
    if not _prompts:
        if not BASTANI_CSV.is_file():
            raise SystemExit(f"No {BASTANI_CSV}: run crutch-test/setup.sh first.")
        log = pd.read_csv(BASTANI_CSV, usecols=["message"], keep_default_na=False)["message"]
        for name, (row, parts, sha256) in PROMPT_RECIPES.items():
            template = rebuild(log.iloc[row], parts)
            if hashlib.sha256(template.encode()).hexdigest() != sha256:
                raise SystemExit(f"{name} rebuilt from {BASTANI_CSV} differs from the pilots' prompt; "
                                 "delete the file and rerun setup.sh.")
            _prompts[name] = template
        assert _prompts["GPT_BASE"].endswith(BASE_PROBLEM_ANCHOR + "{problem}")
    return _prompts


def rebuild(message: str, parts: list[str | tuple[int, int, tuple[int, ...]]]) -> str:
    """Join spans of one logged system turn with our text, putting back the lost newlines."""
    out = []
    for part in parts:
        if isinstance(part, str):  # a placeholder or punctuation of ours
            out.append(part)
            continue
        start, end, newlines = part
        chars = list(message[start:end])
        for i in newlines:
            chars[i] = "\n"
        # Braces in the logged text would read as placeholders in str.format.
        out.append("".join(chars).replace("{", "{{").replace("}", "}}"))
    return "".join(out)


def system_prompt(arm: str, problem: str, answer: str) -> str:
    if arm in ("T", "S"):  # S hands its dialogs to the live tutor after T's solution, with T's prompt
        return tutor_prompts()["GPT_BASE"].format(problem=problem) + EOC_RULE
    if arm == "H":
        return tutor_prompts()["GPT_TUTOR"].format(topic=TOPIC, kind="math", problem=problem,
                                                   answer=answer) + EOC_RULE
    if arm == "W":  # served in-process only; the prefix tells the script which problem it is on
        return WRONG_SYSTEM_PREFIX + problem
    return NULL_SYSTEM_PREFIX + problem


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-heldout", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_parquet(args.test_parquet)
    extra = df["extra_info"]
    rate = extra.map(lambda e: e.get("pre_solve_rate_transfer_problem"))
    has_variant = extra.map(lambda e: bool(e.get("transfer_problem") and e.get("transfer_answer")))
    eligible = [i for i in range(len(df)) if has_variant.iloc[i] and LO <= float(rate.iloc[i]) <= HI]
    print(f"[arms] {len(eligible)}/{len(df)} rows eligible (variant cold solve rate in [1/16, 0.75])")

    order = sorted(eligible, key=lambda i: problem_key(extra.iloc[i]["problem"]))
    random.Random(args.seed).shuffle(order)
    dev = order[: args.n_dev]
    heldout = order[args.n_dev: args.n_dev + args.n_heldout]
    if len(heldout) < args.n_heldout:
        raise SystemExit(f"only {len(heldout)} held-out rows left, need {args.n_heldout}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dev", dev), ("heldout", heldout)):
        sub = df.iloc[rows].reset_index(drop=True)
        for arm in ARMS:
            arm_df = sub.copy()
            arm_df["prompt"] = [
                [
                    {"role": "system", "content": system_prompt(arm, e["problem"], str(e["answer"]))},
                    {"role": "user", "content": OPENER},
                ]
                for e in arm_df["extra_info"]
            ]
            arm_df.to_parquet(out / f"{name}_{arm}.parquet")

    keys = {name: [problem_key(extra.iloc[i]["problem"]) for i in rows]
            for name, rows in (("dev", dev), ("heldout", heldout))}
    content_hash = hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]
    split = {
        "seed": args.seed,
        "eligible": len(eligible),
        "rule": "transfer variant cold solve rate in [1/16, 0.75]",
        "content_hash": content_hash,
        "dev": keys["dev"],
        "heldout": keys["heldout"],
        "dev_mean_pre_solve_rate_transfer": float(rate.iloc[dev].mean()),
        "heldout_mean_pre_solve_rate_transfer": float(rate.iloc[heldout].mean()),
    }
    (out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"[arms] dev {len(dev)} / held-out {len(heldout)}, seed {args.seed}, "
          f"content hash {content_hash}; wrote {', '.join(f'dev_{a}' for a in ARMS)} "
          f"and heldout_* to {out}")


if __name__ == "__main__":
    main()
