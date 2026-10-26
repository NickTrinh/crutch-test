"""Collect the answer tutor's first turns, which the try-first tutor (S) replays.

Usage: python first_turns.py <EduardoRL checkout> <T transfer_conversations_*.jsonl> <out.json>

S shows the student the same solution T gave, after the student's own
attempt, so S - T isolates the attempt. The replay is what T's student saw:
reasoning blocks and <end_of_conversation> are stripped (the tag would end S's
dialog on the spot). Writes {problem key: [T's first turns, by dialog]}.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_routes import problem_key  # noqa: E402


def first_turns(records: list[dict], visible: Callable[[str], str]) -> dict[str, list[str]]:
    """T's first tutor turn per dialog, as its student saw it, grouped by problem."""
    turns: dict[str, list[tuple[int, str]]] = {}
    for rec in records:
        key = problem_key(rec["problem"])
        first = next((m["content"] for m in rec["raw_messages"] if m["role"] == "teacher"), "")
        text = visible(first).strip()
        if not text:  # S would show nothing after the attempt, so stop before the S rollout
            raise SystemExit(f"T's first turn on problem {key} is empty once reasoning is stripped")
        turns.setdefault(key, []).append((int(rec.get("sample_index", 0)), text))
    return {k: [text for _, text in sorted(v)] for k, v in turns.items()}


def main() -> None:
    eduardo, src, out = sys.argv[1:4]
    sys.path.insert(0, str(Path(eduardo).resolve()))
    from eduardo.utils import strip_teacher_controls

    records = [json.loads(line) for line in open(src, encoding="utf-8")]
    # API tutors' replies are visible text unless tagged (see run_eduardo_eval.py).
    turns = first_turns(records, lambda text: strip_teacher_controls(text, assume_thinking=False))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=1, ensure_ascii=False)
    words = [len(t.split()) for texts in turns.values() for t in texts]
    print(f"[first turns] {len(turns)} problems, {len(words)} T dialogs, median {statistics.median(words):.0f} "
          f"words -> {out}")


if __name__ == "__main__":
    main()
