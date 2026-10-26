"""Checks of arms S and W without the network.

  1. every scripted tutor finds its problem in the system prompt make_arms.py writes;
  2. make_arms.py writes all five arms and keeps the proposal's split hash;
  3. S's tutor: the attempt prompt, T's first turns in order, then the live tutor;
  4. W's tutor: the wrong solution, then N's filler and turn count;
  5. a missing replay text fails the dialog instead of falling back;
  6. first_turns.py replays what T's student saw, in dialog order;
  7. wrong_solutions.py keeps only solutions graded wrong that do not give the
     mistake away, and catches correct answers written another way;
  8. make_arms.rebuild puts back the newlines the trial's log lost and keeps
     literal braces out of str.format.

Usage: crutch-test/.venv/bin/python crutch-test/tests/test_arms.py [EduardoRL checkout]
(default: crutch-test/.deps/EduardoRL, which setup.sh clones; 2 needs its .data/test.parquet)
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
EDUARDO = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / ".deps" / "EduardoRL").resolve()
sys.path.insert(0, str(EDUARDO))

import pandas as pd  # noqa: E402

import first_turns  # noqa: E402
import make_arms  # noqa: E402
import openrouter_routes as routes  # noqa: E402
import wrong_solutions  # noqa: E402

PROBLEM = "Let $f(x) = {x}^2$. What is $f(3)$?\n\nGive {an} integer."  # braces and newlines on purpose
KEY = routes.problem_key(PROBLEM)


def reset(**env: str) -> None:
    """Fresh in-process tutor state, reading the given JSON files."""
    routes._maps.clear()
    routes._replayed.clear()
    routes._null_turns = None
    for name in ("CRUTCH_T_FIRST_TURNS", "CRUTCH_WRONG_SOLUTIONS", "CRUTCH_NULL_TURNS", "CRUTCH_LIVE_TUTOR"):
        os.environ.pop(name, None)
    os.environ.update(env)


def write_json(tmp: str, name: str, data) -> str:
    path = Path(tmp) / name
    path.write_text(json.dumps(data))
    return str(path)


def dialog(arm: str, *turns: str) -> list[dict]:
    """The tutor's view: system prompt, then student (user) and tutor (assistant) turns."""
    msgs = [{"role": "system", "content": make_arms.system_prompt(arm, PROBLEM, "9")},
            {"role": "user", "content": make_arms.OPENER}]
    for i, text in enumerate(turns):
        msgs.append({"role": "assistant" if i % 2 == 0 else "user", "content": text})
    return msgs


def tutor_says(model: str, messages: list[dict]) -> str:
    resp = asyncio.run(routes._routed_create(None, model=model, messages=messages, temperature=0.6, top_p=0.95))
    return resp.choices[0].message.content


def test_problem_round_trip() -> None:
    for arm in ("N", "T", "S", "W"):
        assert routes.problem_of(dialog(arm)) == PROBLEM, arm


def test_make_arms() -> None:
    data = HERE / ".data" / "test.parquet"
    if not data.is_file():
        print("  skipped: no .data/test.parquet (run setup.sh)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["make_arms.py", "--test-parquet", str(data), "--out-dir", tmp]
        make_arms.main()
        split = json.loads((Path(tmp) / "split.json").read_text())
        assert split["content_hash"] == "0d1333aed481da89", split["content_hash"]  # the proposal's split
        for name in ("dev", "heldout"):
            frames = {a: pd.read_parquet(Path(tmp) / f"{name}_{a}.parquet") for a in make_arms.ARMS}
            for i, e in enumerate(frames["S"]["extra_info"]):
                prompts = {a: frames[a]["prompt"].iloc[i] for a in frames}
                assert prompts["S"][0]["content"] == prompts["T"][0]["content"]  # S's live tutor is T's
                for a in ("N", "S", "W"):
                    assert routes.problem_of([prompts[a][0]]) == e["problem"], (name, a, i)
                assert all(p[1]["content"] == make_arms.OPENER for p in prompts.values())


def test_try_first_tutor() -> None:
    forwarded = []

    async def fake_create(self, *args, **kwargs):  # stands in for the network
        forwarded.append(kwargs)
        return routes._local_reply(kwargs["model"], 0, "live reply")

    original = routes._original_create
    routes._original_create = fake_create
    try:
        with tempfile.TemporaryDirectory() as tmp:
            reset(CRUTCH_T_FIRST_TURNS=write_json(tmp, "t.json", {KEY: ["T's solution 0", "T's solution 1"]}),
                  CRUTCH_LIVE_TUTOR="anthropic/live-test")
            assert tutor_says(routes.TRY_FIRST_TUTOR, dialog("S")) == routes.TRY_FIRST_TEXT
            attempt = dialog("S", routes.TRY_FIRST_TEXT, "My attempt: 3^2 = 6.")
            # A problem's S dialogs take T's dialogs' first turns in turn.
            assert [tutor_says(routes.TRY_FIRST_TUTOR, attempt) for _ in range(3)] == \
                ["T's solution 0", "T's solution 1", "T's solution 0"]
            assert not forwarded
            later = dialog("S", routes.TRY_FIRST_TEXT, "My attempt.", "T's solution 0", "Why square it?")
            assert tutor_says(routes.TRY_FIRST_TUTOR, later) == "live reply"
            sent = forwarded[-1]
            assert sent["model"] == "anthropic/live-test"
            assert "temperature" not in sent and "top_p" not in sent  # Claude 5.5 takes neither
            assert sent["extra_body"] == routes.CLAUDE_EXTRA_BODY
            assert sent["messages"][0]["content"] == make_arms.system_prompt("T", PROBLEM, "9")
            reset(CRUTCH_T_FIRST_TURNS=write_json(tmp, "none.json", {}))
            assert tutor_says(routes.TRY_FIRST_TUTOR, attempt) == ""  # EduardoRL fails this dialog
    finally:
        routes._original_create = original
        reset()


def test_wrong_tutor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reset(CRUTCH_WRONG_SOLUTIONS=write_json(tmp, "w.json", {"solutions": {KEY: {"text": "f(3) = 6."}}}),
              CRUTCH_NULL_TURNS=write_json(tmp, "turns.json", {KEY: 3}))
        assert tutor_says(routes.WRONG_TUTOR, dialog("W")) == "f(3) = 6."
        assert tutor_says(routes.WRONG_TUTOR, dialog("W", "f(3) = 6.", "So 6?")) == routes.NULL_TEXT
        end = tutor_says(routes.WRONG_TUTOR, dialog("W", "f(3) = 6.", "So 6?", routes.NULL_TEXT, "6."))
        assert end == f"{routes.NULL_TEXT} {routes.END_OF_CONVERSATION}"  # N's turn count, here 3
        reset(CRUTCH_WRONG_SOLUTIONS=write_json(tmp, "none.json", {"solutions": {}}))
        assert tutor_says(routes.WRONG_TUTOR, dialog("W")) == ""
    reset()


def test_first_turns() -> None:
    from eduardo.utils import strip_teacher_controls

    def visible(text):
        return strip_teacher_controls(text, assume_thinking=False)

    def rec(index, tutor):
        return {"problem": PROBLEM, "sample_index": index,
                "raw_messages": [{"role": "student", "content": "What is the answer?"},
                                 {"role": "teacher", "content": tutor}]}

    got = first_turns.first_turns([rec(1, "<think>plan</think>Second. <end_of_conversation>"),
                                   rec(0, "<think>plan</think>First.")], visible)
    assert got == {KEY: ["First.", "Second."]}, got
    try:
        first_turns.first_turns([rec(0, "<think>only reasoning</think>")], visible)
    except SystemExit:
        pass
    else:
        raise AssertionError("an empty first turn must stop the S step")


def test_wrong_solution_check() -> None:
    cases = [
        ("First, square 3 to get 6.\n\nFinal answer: $\\boxed{6}$", "9", None),
        ("First, square 3.\n\nFinal answer: $\\boxed{9}$", "9", "final answer is correct"),
        ("So the result is one half.\n\nFinal answer: $\\boxed{\\frac{1}{2}}$", "0.5", "final answer is correct"),
        ("Final answer: $\\boxed{\\sqrt3}$", "\\sqrt{3}", "rejected"),  # same value, other spelling
        ("The answer is 6.", "9", "no boxed answer"),
        ("First, 3 squared is 6. Wait, that is a mistake.\n\nFinal answer: $\\boxed{6}$", "9", "hints at its mistake"),
        ("", "9", "empty reply"),
    ]
    for text, answer, want in cases:
        why, _ = wrong_solutions.check(text, answer)
        assert (why is not None) if want == "rejected" else (why == want), (text, answer, why)


def test_rebuild() -> None:
    turn = '"Say hinto {x}: done"'  # as logged: "\n" kept as "n", and a literal brace pair
    template = make_arms.rebuild(turn, [(1, 15, (6,)), " {name}"])
    assert template.format(name="Bo") == "Say hi\nto {x}: Bo", template


def main() -> None:
    for test in (test_problem_round_trip, test_make_arms, test_try_first_tutor, test_wrong_tutor,
                 test_first_turns, test_wrong_solution_check, test_rebuild):
        test()
        print(f"ok {test.__name__}")


if __name__ == "__main__":
    main()
