"""Checks of analyze.py's statistics on small synthetic runs; no network.

  1. several dialogs per problem are averaged before the bootstrap, in the
     views table as well as the base tables;
  2. N's turn target is the rounded mean of H's dialogs on that problem;
  3. Holm's step-down adjustment (running maximum, cap at 1);
  4. the bootstrap p-values agree with the interval verdicts they stand for;
  5. the scorecard on a synthetic run with L: P1-P3 Holm-corrected, P4 a
     point rule outside the family;
  6. W's manipulation check counts answers that repeat the planted wrong one.

Usage: crutch-test/.venv/bin/python crutch-test/tests/test_analyze.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze  # noqa: E402
import match_turns  # noqa: E402
from openrouter_routes import problem_key  # noqa: E402


def write_view(run: Path, arm: str, view: str, dialogs: list[dict]) -> None:
    (run / "views").mkdir(parents=True, exist_ok=True)
    (run / "views" / f"{arm}_{view}.json").write_text(json.dumps(
        {"arm": arm, "view": view, "samples": 8, "source": "none", "dialogs": dialogs}))


def test_per_problem() -> None:
    rows = [{"problem": "a", "score": 1.0}, {"problem": "a", "score": 0.5},
            {"problem": "b", "score": 0.25}, {"problem": "b", "score": None}]
    assert analyze.per_problem(rows, "score") == {"a": 0.75, "b": 0.25}


def test_views_average_dialogs() -> None:
    """Two dialogs per problem: the views table must use both, not the last one."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        scores = {"N": [(0.0, 0.0), (0.5, 0.5)], "H": [(0.0, 1.0), (0.5, 0.5)], "T": [(1.0, 0.0), (0.5, 0.0)]}
        for arm, by_problem in scores.items():
            write_view(run, arm, "C", [{"problem": p, "score": s, "removed": 0, "sentences": 1}
                                       for p, pair in zip("ab", by_problem) for s in pair])
        _, out = analyze.views_section(run, set(), np.random.default_rng(0))
        assert out is not None, "the C view should be reported"
        means = out["views"]["C"]["means"]
        # Per-problem means first: T = (0.5 + 0.25) / 2; keeping the last dialog would give 0.0.
        assert means == {"N": 0.25, "H": 0.5, "T": 0.375}, means
        assert out["views"]["C"]["T-N"]["n"] == 2  # problems, not dialogs


def test_turn_targets() -> None:
    recs = [{"problem": "p1", "num_teacher_turns": 3}, {"problem": "p1", "num_teacher_turns": 6},
            {"problem": "p2", "num_teacher_turns": 5}, {"problem": "p3", "num_teacher_turns": 4},
            {"problem": "p3", "num_teacher_turns": 7}]
    got = match_turns.turn_targets(recs)
    # 4.5 and 5.5 round half to even: unbiased across problems.
    assert got == {problem_key("p1"): 4, problem_key("p2"): 5, problem_key("p3"): 6}, got


def test_holm() -> None:
    # b's raw 0.04 times 1 is below c's adjusted 0.06, so the running max lifts it.
    assert analyze.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert analyze.holm({"x": 0.6, "y": 0.5}) == {"y": 1.0, "x": 1.0}  # capped at 1
    assert analyze.holm({"only": 0.02}) == {"only": 0.02}  # one rule: no correction
    assert analyze.holm({}) == {}


def test_p_values_match_intervals() -> None:
    """p < 0.05 implies the interval verdict, which implies p <= 0.05 (equal only on a tie at the cut)."""
    rng = np.random.default_rng(1)
    for trial in range(300):
        n = int(rng.integers(5, 80))
        shift = float(rng.normal(0, 0.06))
        # Scores on a 1/16 grid, as a 16-sample post-test gives: ties are common.
        a = {str(i): float(rng.integers(0, 17)) / 16 for i in range(n)}
        b = {k: min(1.0, max(0.0, v - shift + float(rng.integers(-4, 5)) / 16)) for k, v in a.items()}
        c = analyze.paired(a, b, np.random.default_rng(trial))
        assert c is not None
        for p, ok in ((c["p_above_0"], c["lo"] > 0), (c["p_below_0"], c["hi"] < 0),
                      (c["p_above_margin"], c["lo"] > -analyze.P_MARGIN)):
            assert not (p < 0.05) or ok, (trial, p, c)
            assert not ok or p <= 0.05, (trial, p, c)


def write_learner(run: Path, unrelated_gap: float) -> None:
    """Views L for N, H and T on 200 problems plus the frozen model's scores."""
    rng = np.random.default_rng(2)
    keys = [f"k{i}" for i in range(200)]
    base = {k: {"variant": 0.4, "same": 0.5, "unrelated": 0.3} for k in keys}
    shift = {"N": (0.0, 0.0), "H": (0.0, 0.0), "T": (-0.1, unrelated_gap)}  # (variant, unrelated)
    for arm, (dv, du) in shift.items():
        dialogs = [{"problem": k, "score": 0.4 + dv + float(rng.normal(0, 0.05)), "same": 0.5,
                    "unrelated": 0.3 + du + float(rng.normal(0, 0.05)),
                    "n_train_tokens": int(rng.integers(300, 900)), "drop": 0.2}
                   for k in keys for _ in range(2)]  # two dialogs per problem
        write_view(run, arm, "L", dialogs)
    (run / "views" / "base_L.json").write_text(json.dumps(
        {"lr": 0.1, "base": base, "calibration": {"0.1": [0.25]}}))


def test_scorecard_with_learner() -> None:
    practice = analyze.paired({f"k{i}": 0.9 for i in range(200)}, {f"k{i}": 0.3 for i in range(200)},
                              np.random.default_rng(3))
    for gap, p4_pass in ((0.01, True), (0.06, False)):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_learner(run, gap)
            _, out = analyze.views_section(run, set(), np.random.default_rng(0))
            assert out is not None and "learner" in out
            tests = {"L": {"T-N": out["views"]["L"]["T-N"], "H-N": out["views"]["L"]["H-N"]}}
            lines, card = analyze.scorecard(practice, tests, out["learner"]["P4"]["T-N"])
            rules = card["L"]
            assert set(rules) == {"P1", "P2", "P3", "P4"}, rules
            # The family is P1-P3: their Holm p-values are holm() of their raw p-values.
            raw = {r: rules[r]["p"] for r in ("P1", "P2", "P3")}
            assert {r: rules[r]["p_holm"] for r in raw} == analyze.holm(raw)
            assert rules["P2"]["pass"] and rules["P2"]["pass_holm"], rules["P2"]  # T - N = -0.1, tight
            assert rules["P3"]["pass"] and rules["P3"]["pass_holm"], rules["P3"]  # H - N about 0
            assert rules["P4"]["p_holm"] is None and rules["P4"]["pass"] is p4_pass is rules["P4"]["pass_holm"]
            assert any(line.startswith("| L | P4 ") and "n/a (point rule)" in line for line in lines)


def test_holm_can_flip_a_verdict() -> None:
    """A rule that passes alone can fail once corrected for three rules."""
    c = {"n": 50, "mean": -0.05, "lo": -0.09, "hi": -0.004, "sd": 0.1,
         "p_above_0": 1.0, "p_below_0": 0.03, "p_above_margin": 0.9}
    practice = {"n": 50, "mean": 0.5, "lo": 0.4, "hi": 0.6, "sd": 0.1,
                "p_above_0": 0.0, "p_below_0": 1.0, "p_above_margin": 0.0}
    _, card = analyze.scorecard(practice, {"M": {"T-N": c, "H-N": c}}, None)
    p2 = card["M"]["P2"]
    assert p2["pass"] and not p2["pass_holm"] and abs(p2["p_holm"] - 0.06) < 1e-12, p2


def test_w_copy_check() -> None:
    """W's manipulation check: how often its student repeats the planted wrong answer on view P."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "wrong_solutions.json").write_text(json.dumps(
            {"problems": 2, "solutions": {"k1": {"pred": "7"}, "k2": {"pred": "\\frac{1}{2}"}}, "dropped": {}}))
        rows = [{"tag": "W_open", "test": "same", "problem": "k1", "pred": p} for p in ("7", " 7", "9", "")] + \
               [{"tag": "W_open", "test": "same", "problem": "k2", "pred": "$\\frac{1}{2}$"},
                {"tag": "W_open", "test": "variant", "problem": "k1", "pred": "7"},  # another test: not counted
                {"tag": "N_open", "test": "same", "problem": "k1", "pred": "7"}]     # another arm: not counted
        (run / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        level = {a: {v: {"k1": 0.5, "k2": 0.5} for v in analyze.VIEWS} for a in ("N", "H", "T", "W")}
        arms = {a: {"handover_rate": 0.0, "posed_new_problem": 0} for a in level}
        lines, out = analyze.extra_arms_section(run, level, arms, np.random.default_rng(1))
        assert out is not None and out["wrong_solutions"]["copied_on_P"] == 0.6, out  # 3 of 5
        assert any("60% of W's answers on the same problem repeat the planted wrong answer" in x for x in lines)
        assert not any("teaching judge" in x for x in lines)


def main() -> None:
    for test in (test_per_problem, test_views_average_dialogs, test_turn_targets, test_holm,
                 test_p_values_match_intervals, test_scorecard_with_learner, test_holm_can_flip_a_verdict,
                 test_w_copy_check):
        test()
        print(f"ok {test.__name__}")


if __name__ == "__main__":
    main()
