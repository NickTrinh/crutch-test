"""Give the no-AI control the same number of tutor turns the H arm used.

Usage: python match_turns.py <H transfer_conversations_*.jsonl> <out.json>

Writes {problem key: tutor turns}; crutch/null-tutor reads it through
CRUTCH_NULL_TURNS and ends each dialog on the matching turn.
"""

import collections
import json
import statistics
import sys

from openrouter_routes import problem_key


def turn_targets(records: list[dict]) -> dict[str, int]:
    """Tutor turns per problem: the mean over its dialogs, rounded half to even.

    With several dialogs per problem the null tutor cannot tell them apart
    (they share one system prompt), so every dialog of a problem gets the mean.
    """
    turns: dict[str, list[int]] = {}
    for rec in records:
        turns.setdefault(problem_key(rec["problem"]), []).append(int(rec["num_teacher_turns"]))
    return {k: round(statistics.fmean(v)) for k, v in turns.items()}


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    records = [json.loads(line) for line in open(src, encoding="utf-8")]
    turns = turn_targets(records)
    ends = collections.Counter(rec.get("end_reason") for rec in records)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=1)
    print(f"[turns] {len(turns)} problems ({len(records)} dialogs), mean {statistics.fmean(turns.values()):.2f} "
          f"tutor turns (end reasons {dict(ends)}) -> {out}")


if __name__ == "__main__":
    main()
