"""Turn a pilot run into the proposal's go/no-go table and first results.

Usage: python analyze.py runs/pilot50 [--report results/pilot50.md]

Views (what the student carries into its unaided post-test):
  P   same problem, full dialog visible      (practice; human target: up)
  U   new variant, full dialog visible
  M   new variant, tutor turns hidden        (EduardoRL's test; human target: down)
  Ms  same problem, tutor turns hidden
Every contrast is paired by problem against the no-AI arm N, with a 95%
cluster-bootstrap interval over problems.

If views.py has run, the report adds the views it scores on the same dialogs
(M rescored, C copy-filtered, K carried note) with the proposal's pass rules.
If learner_grade.py has run, it adds L, the student that learns from its own
turns, with checks that it learned (vs the frozen model) and did not drift (P4).
The pass rules P1-P4 are then scored per test view, unadjusted and with Holm's
correction over each view's rules.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

import numpy as np

from openrouter_routes import JUDGE, STUDENT, problem_key

VIEWS = {
    "P": ("open", "post_solve_same"),
    "U": ("open", "post_solve_transfer"),
    "M": ("masked", "post_solve_transfer"),
    "Ms": ("masked", "post_solve_same"),
}
ARMS = ("N", "H", "T")
# Optional arms: S tries first, then sees T's solution; W gets a confidently
# wrong solution (the positive control). Neither enters the pass rules.
EXTRA_ARMS = ("S", "W")
# Full Crutch Test plan (proposal, compute table): 201,600 student answers on 300 problems.
AUDIT_ANSWERS = 201_600
GPU_TOKENS_PER_S = 2_500
GATE_PRACTICE, GATE_LEAK_T, GATE_LEAK_H, GATE_PRETEST, GATE_GPU_H = 0.20, 0.80, 0.20, 0.05, 12.0


ATTEMPTS: dict[str, int] = {}  # tag -> student post-test calls requested
MAX_FAILED = 0.01  # above this share of failed student calls the run is not usable
# What pilot.sh ran before it recorded a run's design in runs/<run>/config.
LEGACY_DESIGN = "split=dev dialogs=1 arms=N,H,T"


def run_design(run: Path) -> dict[str, str]:
    """The design pilot.sh recorded for this run folder, as {"split": ..., ...}."""
    path = run / "config"
    text = path.read_text(encoding="utf-8") if path.is_file() else LEGACY_DESIGN
    return dict(tok.split("=", 1) for tok in text.split())


def load(run: Path, tag: str) -> list[dict]:
    files = sorted(glob.glob(str(run / "eval" / f"transfer_eval_{tag}_*.json")))
    if not files:
        raise SystemExit(f"missing results for {tag} in {run}/eval")
    data = json.load(open(files[-1], encoding="utf-8"))
    # Each conversation gets K attempts on the variant and K on the original.
    ATTEMPTS[tag] = len(data["conversations"]) * 2 * int(data["args"]["num_student_attempts"])
    return data["conversations"]


def by_problem(records: list[dict], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in records:
        groups.setdefault(problem_key(r["problem"]), []).append(float(r[field]))
    return {k: statistics.fmean(v) for k, v in groups.items()}


def per_problem(dialogs: list[dict], field: str) -> dict[str, float]:
    """Mean of field over each problem's dialogs, for rows keyed by problem key.

    A run can have several dialogs per problem; the bootstrap resamples
    problems, so their dialogs are averaged first. Missing scores are skipped.
    """
    groups: dict[str, list[float]] = {}
    for x in dialogs:
        if x.get(field) is not None:
            groups.setdefault(x["problem"], []).append(float(x[field]))
    return {k: statistics.fmean(v) for k, v in groups.items()}


def paired(a: dict[str, float], b: dict[str, float], rng: np.random.Generator, n_boot: int = 10_000):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    if len(d) == 0:
        return None
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Bootstrap p-values for the pass rules, from the same resamples as the
    # interval (no new draws, so every interval stays as it was): twice the
    # share of means on the far side of the rule's boundary, so p < 0.05 is
    # the 95% interval clearing it. Ties count against passing.
    def p_past(side_share: float) -> float:
        return float(min(1.0, 2 * side_share))

    return {"n": len(d), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "p_above_0": p_past(np.mean(boot <= 0)),            # P1: practice rises
            "p_below_0": p_past(np.mean(boot >= 0)),            # P2: the test falls
            "p_above_margin": p_past(np.mean(boot <= -P_MARGIN)),  # P3: no harm beyond the margin
            }


def fmt(c) -> str:
    return "n/a" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


# A tutor that sets the student a new "similar problem" gives in-dialog
# near-transfer practice, which can raise the variant post-test on its own.
# Explicit phrasings only: "Try this:" also introduces hint steps.
POSED = re.compile(
    r"here(?:'s| is) (?:a|one|another)\b[^.?\n]{0,40}\b(?:similar|practice|new) problem\b"
    r"|(?:similar|practice|new) problem (?:to try|for you)"
    r"|\*\*problem:\*\*|good luck with the similar problem", re.I)


def posed_new_problem(rec: dict) -> bool:
    return any(POSED.search(re.sub(r"<think>.*?</think>", "", m["content"], flags=re.S))
               for m in rec["raw_messages"] if m["role"] == "teacher")


P_MARGIN = 0.03  # the proposal's pass rules use a fixed 0.03 margin
CARRIES = {
    "M": "student turns; tutor turns hidden",
    "C": "M minus sentences copied from the tutor",
    "K": "one note of at most 120 tokens",
    "L": "weights trained on its own turns; no transcript",
}


def _norm(text: str) -> str:
    return re.sub(r"[\s$\\{}]", "", str(text)).lower()


def _answer_rows(run: Path) -> list[dict]:
    path = run / "answers.jsonl"
    return list(map(json.loads, open(path, encoding="utf-8"))) if path.is_file() else []


def views_section(run: Path, posed_t: set, rng: np.random.Generator, pilot_m: dict | None = None):
    """Report views.py results; returns (markdown lines, data) or ([], None).

    When views.py did not rescore M, the pilot's own hidden pass stands in for
    it, provided its answers were logged at the views' 2,048-token cap.
    """
    vdir = run / "views"
    data = {}
    for f in sorted(vdir.glob("*_*.json")) if vdir.is_dir() else []:
        if f.name == "base_L.json":  # the frozen model's scores, read by learner_section
            continue
        d = json.load(open(f, encoding="utf-8"))
        data[(d["arm"], d["view"])] = d
    rows = _answer_rows(run)
    m_tags = {}  # arm -> answer-log tag of view M
    pilot_caps = {r.get("cap") for r in rows if r.get("tag", "").endswith("_masked") and r.get("test") == "variant"}
    if pilot_m and any((a, v) in data for a in ARMS for v in CARRIES) \
            and not any((a, "M") in data for a in ARMS) and pilot_caps == {2048}:
        for a in ARMS:
            data[(a, "M")] = {"arm": a, "view": "M", "source": "pilot",
                              "dialogs": [{"problem": k, "score": x} for k, x in pilot_m[a].items()]}
            m_tags[a] = f"{a}_masked"
    views = [v for v in CARRIES if all((a, v) in data for a in ARMS)]
    if not views:
        return [], None
    score = {key: per_problem(d["dialogs"], "score") for key, d in data.items()}
    out = {"views": {}, "split": {}, "diagnostics": {}}
    lines = ["", "## More simulated students on the same dialogs (views.py)", "",
             "Each view changes only what the student carries into its test on the new variant. "
             "Answers are capped at 2,048 tokens and all logged. Human target from Bastani's exam: "
             "T - N about -0.05 (GPT Base -17%), H - N about 0 (GPT Tutor).", "",
             "| View | Carries | N | H | T | H - N | T - N | Test down (P2) | Hints safe (P3) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in views:
        hn = paired(score[("H", v)], score[("N", v)], rng)
        tn = paired(score[("T", v)], score[("N", v)], rng)
        p2 = tn is not None and tn["mean"] <= -P_MARGIN and tn["hi"] < 0
        p3 = hn is not None and hn["lo"] > -P_MARGIN
        means = {a: statistics.fmean(score[(a, v)].values()) for a in ARMS}
        out["views"][v] = {"means": means, "H-N": hn, "T-N": tn, "P2": p2, "P3": p3}
        lines.append(f"| {v} | {CARRIES[v]} | {means['N']:.3f} | {means['H']:.3f} | {means['T']:.3f} | "
                     f"{fmt(hn)} | {fmt(tn)} | {'pass' if p2 else 'fail'} | {'pass' if p3 else 'fail'} |")
    lines += ["", "P2 passes when T - N <= -0.03 with its interval below 0; P3 when the lower end of H - N is above -0.03.",
              "", "| View | T - N where T set a new problem | T - N where it did not |", "|---|---|---|"]
    for v in views:
        yes = paired({k: x for k, x in score[("T", v)].items() if k in posed_t}, score[("N", v)], rng)
        no = paired({k: x for k, x in score[("T", v)].items() if k not in posed_t}, score[("N", v)], rng)
        out["split"][v] = {"posed": yes, "not_posed": no}
        lines.append(f"| {v} | {fmt(yes)} (n={yes['n'] if yes else 0}) | {fmt(no)} (n={no['n'] if no else 0}) |")

    if all((a, "C") in data for a in ARMS):
        removed = {a: (sum(x.get("removed", 0) for x in data[(a, "C")]["dialogs"]),
                       sum(x.get("sentences", 0) for x in data[(a, "C")]["dialogs"])) for a in ARMS}
        out["c_removed"] = removed
        lines += ["", "C removed " + ", ".join(f"{a} {r}/{t} ({r / max(1, t):.0%})" for a, (r, t) in removed.items())
                  + " of student sentences as copied from the tutor."]
    if all((a, "K") in data for a in ARMS):
        stats = {}
        for a in ARMS:
            src = run / "eval" / data[(a, "K")]["source"]
            answer = {problem_key(r["problem"]): r["answer"] for r in map(json.loads, open(src, encoding="utf-8"))}
            notes = [(x["problem"], x.get("note") or "") for x in data[(a, "K")]["dialogs"]]
            stats[a] = {"words": statistics.fmean(len(n.split()) for _, n in notes),
                        "has_answer": statistics.fmean(_norm(answer[k]) in _norm(n) for k, n in notes)}
        out["k_notes"] = stats
        lines += ["", "K notes: " + ", ".join(f"{a} {s['words']:.0f} words, {s['has_answer']:.0%} contain the original answer"
                                              for a, s in stats.items()) + "."]

    if m_tags:
        lines += ["", "M here is the pilot's own hidden pass (same prompts, same 2,048-token cap), "
                  "so views.py did not rescore it."]
    if rows:
        rows = [r for r in rows if r.get("test", "variant") == "variant"
                and (r.get("view") in views or r.get("tag") in m_tags.values())]
        lines += ["", "| View | Arm | Answers | Cut off at the cap | No boxed answer | Correct among boxed |", "|---|---|---|---|---|---|"]
        for v in views:
            for a in ARMS:
                tag = m_tags[a] if v == "M" and m_tags else f"{a}_{v}"
                rs = [r for r in rows if r["tag"] == tag]
                if not rs:
                    continue
                boxed = [r for r in rs if not r["unboxed"]]
                diag = {"answers": len(rs), "cut_off": statistics.fmean(r["finish"] == "length" for r in rs),
                        "unboxed": statistics.fmean(r["unboxed"] for r in rs),
                        "acc_boxed": statistics.fmean(r["acc"] for r in boxed) if boxed else None}
                out["diagnostics"][f"{a}_{v}"] = diag
                acc = "n/a" if diag["acc_boxed"] is None else f"{diag['acc_boxed']:.3f}"
                lines.append(f"| {v} | {a} | {diag['answers']:,} | {diag['cut_off']:.1%} | {diag['unboxed']:.1%} | {acc} |")
    if all((a, "L") in data for a in ARMS) and (vdir / "base_L.json").is_file():
        l_lines, out["learner"] = learner_section(data, json.load(open(vdir / "base_L.json")), rng)
        lines += l_lines
    vledger = vdir / "ledger.jsonl"
    if vledger.is_file():
        usd = sum(m["cost_usd"] for row in map(json.loads, open(vledger)) for m in row["models"].values())
        out["cost_usd"] = usd
        lines += ["", f"Cost of these views: ${usd:.2f}."]
    return lines, out


def _ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r = np.empty(len(x))
    r[x.argsort(kind="mergesort")] = np.arange(len(x))
    for v in np.unique(x):  # ties share their average rank
        r[x == v] = r[x == v].mean()
    return r


def learner_section(data: dict, base_l: dict, rng: np.random.Generator):
    """View L: did the student learn from its turns, did it drift, and does volume matter?"""
    base = base_l["base"]
    dialogs = {a: data[(a, "L")]["dialogs"] for a in ARMS}
    out = {"lr": base_l["lr"], "arms": {}}
    lines = ["", "### The learning student (L)", "",
             f"Each dialog trains its own LoRA adapter on the student's own turns (plain SGD, lr {base_l['lr']}, "
             "picked on N and H dialogs only); the test then shows no transcript. "
             "\"Frozen\" is the same model with no training, on the same tests.", "",
             "| Arm | Trained tokens (median) | Loss drop (median) | Variant: L - frozen | Same problem: L - frozen "
             "| Unrelated: L - frozen |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        ds = dialogs[a]
        gain = {kind: paired(per_problem(ds, field),
                             {k: b[kind] for k, b in base.items() if b.get(kind) is not None}, rng)
                for kind, field in (("variant", "score"), ("same", "same"), ("unrelated", "unrelated"))}
        tokens = statistics.median(x["n_train_tokens"] for x in ds)
        drop = statistics.median(x["drop"] for x in ds)
        out["arms"][a] = {"median_train_tokens": tokens, "median_loss_drop": drop, "vs_frozen": gain}
        lines.append(f"| {a} | {tokens:,.0f} | {drop:.1%} | {fmt(gain['variant'])} | {fmt(gain['same'])} "
                     f"| {fmt(gain['unrelated'])} |")
    unrelated = {a: per_problem(dialogs[a], "unrelated") for a in ARMS}
    p4 = paired(unrelated["T"], unrelated["N"], rng)
    out["P4"] = {"T-N": p4, "pass": p4 is not None and abs(p4["mean"]) <= P_MARGIN}
    pairs = [(x["n_train_tokens"], x["score"] - base[x["problem"]]["variant"]) for a in ARMS for x in dialogs[a]
             if x["score"] is not None and base.get(x["problem"], {}).get("variant") is not None]
    rho = float(np.corrcoef(_ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs]))[0, 1]) \
        if len(pairs) > 2 else None
    out["tokens_vs_gain_spearman"] = rho
    calib = ", ".join(f"{lr}: {statistics.median(d):.1%}" for lr, d in base_l["calibration"].items() if d)
    minutes = ", ".join(f"{k.replace('_minutes', '')} {v:.0f} min" for k, v in base_l.get("minutes", {}).items())
    lines += ["", f"Drift check (P4): on an unrelated problem, T - N = {fmt(p4)}, "
              f"{'pass' if out['P4']['pass'] else 'fail'} (rule: within +/-{P_MARGIN}). "
              "A shift here means training changed the student in general, not what it practised.",
              "", "Does practice volume matter? Spearman correlation across all dialogs between trained tokens "
              f"and the variant gain over frozen: {'n/a' if rho is None else f'{rho:+.2f}'}.",
              "", f"Learning-rate calibration (median loss drop on N and H): {calib}."
              + (f" GPU time: {minutes}." if minutes else "")]
    return lines, out


def extra_arms_section(run: Path, level: dict, arms: dict, rng: np.random.Generator):
    """Arms S and W against N and T on every view; returns (markdown lines, data) or ([], None)."""
    extra = [a for a in EXTRA_ARMS if a in level]
    if not extra:
        return [], None
    scores = {v: {a: level[a][v] for a in level} for v in VIEWS}
    for view in ("C", "K"):  # views.py views, when it also scored the extra arms
        files = {a: run / "views" / f"{a}_{view}.json" for a in ("N", "T", *extra)}
        if all(f.is_file() for f in files.values()):
            scores[view] = {a: per_problem(json.load(open(f, encoding="utf-8"))["dialogs"], "score")
                            for a, f in files.items()}
    out = {"contrasts": {}}
    lines = ["", "## Arms S and W", "",
             "S: the tutor asks for an attempt, then shows T's first turn on that problem, then the live GPT Base "
             "tutor carries on, so S - T is the effect of trying first with the same solution shown. W: a "
             "confidently wrong solution, then N's filler, so W - N is the positive control (the proposal: W must "
             "lower scores). Paired by problem with 95% bootstrap intervals; neither arm enters the pass rules.", "",
             "| View | N | T | S | W | S - N | S - T | W - N | W lowers scores |", "|---|---|---|---|---|---|---|---|---|"]
    for v, s in scores.items():
        c = {name: paired(s[a], s[b], rng) if a in s and b in s else None
             for name, (a, b) in (("S-N", ("S", "N")), ("S-T", ("S", "T")), ("W-N", ("W", "N")))}
        lowers = None if c["W-N"] is None else c["W-N"]["hi"] < 0
        out["contrasts"][v] = {**c, "W lowers scores": lowers}
        mean = {a: f"{statistics.fmean(s[a].values()):.3f}" if a in s else "n/a" for a in ("N", "T", "S", "W")}
        lines.append(f"| {v} | {mean['N']} | {mean['T']} | {mean['S']} | {mean['W']} | {fmt(c['S-N'])} | "
                     f"{fmt(c['S-T'])} | {fmt(c['W-N'])} | {'n/a' if lowers is None else ('yes' if lowers else 'NO')} |")
    checks = []
    if "S" in arms:
        checks.append(f"Handover judged in {arms['S']['handover_rate']:.0%} of S dialogs (T: "
                      f"{arms['T']['handover_rate']:.0%}); a solution shown after the student already gave the "
                      f"answer is not a handover. S's tutor set a new practice problem on "
                      f"{arms['S']['posed_new_problem']} problems.")
    wrong = run / "wrong_solutions.json"
    if "W" in arms and wrong.is_file():
        w = json.load(open(wrong, encoding="utf-8"))
        out["wrong_solutions"] = {"used": len(w["solutions"]), "problems": w["problems"], "dropped": w["dropped"]}
        checks.append(f"W: {len(w['solutions'])} of {w['problems']} problems got a wrong solution whose boxed "
                      "answer was graded wrong and which does not hint at its mistake; W runs on those.")
        # The direct manipulation check. The teaching judge is no use here: it
        # already rejects N's repeated filler, which W shares.
        planted = {k: _norm(s["pred"]) for k, s in w["solutions"].items()}
        same = [r for r in _answer_rows(run) if r.get("tag") == "W_open" and r.get("test") == "same"]
        if same:
            copied = statistics.fmean(_norm(r["pred"]) == planted.get(r["problem"]) for r in same)
            out["wrong_solutions"]["copied_on_P"] = copied
            checks.append(f"With the dialog in view (P), {copied:.0%} of W's answers on the same problem repeat the "
                          "planted wrong answer.")
    return lines + [""] + [f"- {c}" for c in checks], out


ALPHA = 0.05
RULES = {  # proposal, Evaluation table
    "P1": "practice up (P: T - N)",
    "P2": "test down (T - N)",
    "P3": "hints safe (H - N)",
    "P4": "no drift (unrelated: T - N)",
}


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjusted p-values.

    The i-th smallest of m p-values is multiplied by m - i + 1 and capped at 1;
    a running maximum keeps the adjusted values in the same order as the raw ones.
    """
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(ps) - i) * p))
        adjusted[name] = running
    return adjusted


def _p(p: float | None) -> str:
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def scorecard(practice: dict | None, tests: dict[str, dict], p4: dict | None):
    """Pass rules P1-P4 per test view, unadjusted and with Holm's correction.

    practice is T - N on view P, which P1 scores for every view; tests maps a
    view to its {"T-N", "H-N"} contrasts; p4 is L's unrelated-problem T - N.
    Each view is one family, Holm-corrected over P1-P3. P4 is a point rule
    (within +/-0.03), so it has no p-value; it stays out of the family, since
    a drift test inside Holm would make passing easier, not harder.
    """
    lines = ["", "## Pass rules P1-P4, unadjusted and with Holm's correction", "",
             f"Each view is one family: Holm's step-down correction over P1-P3 at {ALPHA:.0%}. p is the bootstrap "
             "p-value of the rule's interval (twice the share of resampled means past the boundary), so p < 0.05 "
             "matches the interval verdict. P2 also needs T - N <= -0.03. P4 exists only for L; it is a point "
             "rule (within +/-0.03), reported unadjusted.", "",
             "| View | Rule | Estimate [95% CI] | p | Holm p | Pass | Pass (Holm) |", "|---|---|---|---|---|---|---|"]
    out: dict[str, dict] = {}
    for view, c in tests.items():
        tn, hn = c["T-N"], c["H-N"]
        rules = {  # rule -> (contrast, its p-value field, the pre-registered interval verdict)
            "P1": (practice, "p_above_0", practice is not None and practice["lo"] > 0),
            "P2": (tn, "p_below_0", tn is not None and tn["mean"] <= -P_MARGIN and tn["hi"] < 0),
            "P3": (hn, "p_above_margin", hn is not None and hn["lo"] > -P_MARGIN),
        }
        ps = {r: con[field] for r, (con, field, _) in rules.items() if con is not None}
        adjusted = holm(ps)
        out[view] = {}
        for r, (con, _, ok) in rules.items():
            point_ok = r != "P2" or (con is not None and con["mean"] <= -P_MARGIN)
            ok_holm = r in adjusted and adjusted[r] < ALPHA and point_ok
            out[view][r] = {"estimate": con, "p": ps.get(r), "p_holm": adjusted.get(r), "pass": ok, "pass_holm": ok_holm}
            lines.append(f"| {view} | {r} {RULES[r]} | {fmt(con)} | {_p(ps.get(r))} | {_p(adjusted.get(r))} "
                         f"| {'pass' if ok else 'fail'} | {'pass' if ok_holm else 'fail'} |")
        if view == "L" and p4 is not None:
            ok = abs(p4["mean"]) <= P_MARGIN
            out[view]["P4"] = {"estimate": p4, "p": None, "p_holm": None, "pass": ok, "pass_holm": ok}
            verdict = "pass" if ok else "fail"
            lines.append(f"| L | P4 {RULES['P4']} | {fmt(p4)} | n/a | n/a (point rule) | {verdict} | {verdict} |")
    return lines, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--report", default=None, help="markdown output (default: <run>/report.md)")
    args = ap.parse_args()
    run = Path(args.run)
    rng = np.random.default_rng(0)

    # S and W are optional: reported when both of their post-test passes exist.
    present = list(ARMS) + [a for a in EXTRA_ARMS if all(
        glob.glob(str(run / "eval" / f"transfer_eval_{a}_{p}_*.json")) for p in ("masked", "open"))]
    recs = {(arm, p): load(run, f"{arm}_{p}") for arm in present for p in ("masked", "open")}
    level = {arm: {v: by_problem(recs[(arm, p)], f) for v, (p, f) in VIEWS.items()} for arm in present}
    contrasts = {f"{arm}-N": {v: paired(level[arm][v], level["N"][v], rng) for v in VIEWS}
                 for arm in ("T", "H")}

    arms = {}
    for arm in present:
        masked = recs[(arm, "masked")]
        turns = [r["num_teacher_turns"] for r in masked]
        arms[arm] = {
            "dialogs": len(masked),
            "handover_rate": statistics.fmean(r["leaked"] for r in masked),
            # The quality judge's verdict, kept for the record. It cannot check W:
            # it also rejects N's repeated filler (96% of N dialogs in PILOT-50 v2).
            "teaching_rejected": statistics.fmean(r.get("teaching_rejected", 0) for r in masked),
            "tutor_turns": statistics.fmean(turns),
            "blanked_turns": sum(r.get("num_blanked_teacher_turns", 0) for r in masked) / max(1, sum(turns)),
            "failed_student_calls": sum(r.get("n_student_calls_failed", 0) for p in ("masked", "open")
                                        for r in recs[(arm, p)]),
            "student_calls": sum(ATTEMPTS[f"{arm}_{p}"] for p in ("masked", "open")),
            "failed_dialogs": sum(1 for r in masked if "failed" in str(r.get("end_reason"))),
            "unboxed_answers": sum(r.get("n_incorrect_format", 0) for r in masked),
            "end_reasons": {e: sum(1 for r in masked if r.get("end_reason") == e)
                            for e in sorted({r.get("end_reason") for r in masked})},
            "means": {v: statistics.fmean(level[arm][v].values()) for v in VIEWS},
        }

    posed = {arm: {problem_key(r["problem"]) for r in recs[(arm, "masked")] if posed_new_problem(r)} for arm in present}
    for arm in present:
        arms[arm]["posed_new_problem"] = len(posed[arm])
    split = {}
    for name, keep in (("T posed a new problem", lambda k: k in posed["T"]),
                       ("T did not", lambda k: k not in posed["T"])):
        split[name] = {v: paired({k: x for k, x in level["T"][v].items() if keep(k)}, level["N"][v], rng)
                       for v in ("M", "U")}

    pretest = None
    if (run / "pretest.json").is_file():
        pretest = json.load(open(run / "pretest.json"))["summary"]

    # Cost and projection from the ledger (OpenRouter reports dollars per call).
    ledger = [json.loads(l) for l in open(run / "ledger.jsonl")] if (run / "ledger.jsonl").is_file() else []
    cost = {}
    for row in ledger:
        for model, e in row["models"].items():
            c = cost.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            for k in c:
                c[k] += e[k]
    scoring = [row["models"].get(STUDENT) for row in ledger if row["tag"].endswith(("_masked", "_open"))]
    scoring = [s for s in scoring if s and s["calls"]]
    projection = None
    if scoring:
        calls = sum(s["calls"] for s in scoring)
        out_tok = sum(s["completion_tokens"] for s in scoring) / calls
        usd = sum(s["cost_usd"] for s in scoring) / calls
        projection = {"tokens_per_answer": out_tok, "usd_per_answer": usd,
                      "audit_gpu_hours": AUDIT_ANSWERS * out_tok / GPU_TOKENS_PER_S / 3600,
                      "audit_api_usd": AUDIT_ANSWERS * usd}

    p_tn = contrasts["T-N"]["P"]
    gates = {
        "practice: P(T) - P(N) >= +0.20": None if p_tn is None else p_tn["mean"] >= GATE_PRACTICE,
        "judge: handover T >= 80%": arms["T"]["handover_rate"] >= GATE_LEAK_T,
        "judge: handover H <= 20%": arms["H"]["handover_rate"] <= GATE_LEAK_H,
        "pre-test within 0.05 of the dataset": None if pretest is None else abs(pretest["variant_gap"]) <= GATE_PRETEST,
        "300-problem audit <= 12 A100-hours": None if projection is None
        else projection["audit_gpu_hours"] <= GATE_GPU_H,
    }
    m_tn = contrasts["T-N"]["M"]
    failed = sum(a["failed_student_calls"] for a in arms.values())
    calls = sum(a["student_calls"] for a in arms.values())
    if pretest is not None:
        failed += pretest["failed_calls"]
        calls += pretest["problems"] * 2 * pretest["k"]
    failed_dialogs = sum(a["failed_dialogs"] for a in arms.values())
    if failed > MAX_FAILED * calls or failed_dialogs:
        decision = (f"INVALID RUN: {failed:,} of {calls:,} student calls ({failed / calls:.1%}) and "
                    f"{failed_dialogs} dialogs failed after retries, so scores are biased down. "
                    "Rerun with FORCE=1, a lower CONCURRENCY or more RETRIES.")
    elif any(v is False for v in gates.values()):
        failing = "; ".join(name for name, ok in gates.items() if ok is False)
        decision = (f"NO-GO for now (failing: {failing}). Fix it for up to two days, "
                    "then try the Qwen3-8B student.")
    elif any(v is None for v in gates.values()):
        decision = "INCOMPLETE: a gate could not be computed."
    else:
        decision = "GO."
    if m_tn is not None and m_tn["mean"] <= -0.05:
        framing = ("The masked student already pays for copied answers (M: T-N <= -0.05). "
                   "Pre-registered pivot: the headline becomes which protocol components create the cost.")
    elif m_tn is not None and m_tn["mean"] >= -0.02:
        framing = ("The masked student does not pay for copied answers (M: T-N >= -0.02), "
                   "the gap the project set out to test.")
    else:
        framing = "M: T-N sits between -0.05 and -0.02; the 300-problem run decides."

    design = run_design(run)
    n_problems = len(level["N"]["M"])
    per_arm = len(recs[("N", "masked")]) / max(1, n_problems)
    title = "PILOT report" if design["split"] == "dev" else "Held-out report"
    lines = [f"# {title}: {run.name}", "",
             f"**Split:** {design['split']}, {n_problems} problems, {per_arm:g} dialog{'s' * (per_arm != 1)} per arm.", "",
             f"**Decision:** {decision}", "", f"**Framing:** {framing}", "",
             "## Go/no-go gates (proposal, Risks section)", "", "| Gate | Value | Pass |", "|---|---|---|"]
    values = [
        fmt(p_tn),
        f"{arms['T']['handover_rate']:.0%}",
        f"{arms['H']['handover_rate']:.0%}",
        "n/a" if pretest is None else f"{pretest['variant_gap']:+.3f} (measured {pretest['measured_variant']:.3f} vs stored {pretest['stored_variant']:.3f})",
        "n/a" if projection is None else f"{projection['audit_gpu_hours']:.1f} h ({projection['tokens_per_answer']:.0f} tokens per answer)",
    ]
    for (name, ok), val in zip(gates.items(), values):
        lines.append(f"| {name} | {val} | {'n/a' if ok is None else ('yes' if ok else 'NO')} |")
    lines += ["", "## Post-test solve rate by view (paired by problem vs N; 95% bootstrap CI)", "",
              "| View | N | H | T | H - N | T - N |", "|---|---|---|---|---|---|"]
    for v in VIEWS:
        lines.append(f"| {v} | {arms['N']['means'][v]:.3f} | {arms['H']['means'][v]:.3f} | {arms['T']['means'][v]:.3f}"
                     f" | {fmt(contrasts['H-N'][v])} | {fmt(contrasts['T-N'][v])} |")
    if m_tn is not None:
        lines += ["", f"Paired SD of T - N per problem under M: {m_tn['sd']:.3f} (the power analysis assumed 0.18)."]
    lines += ["", "## Did the answer tutor set new problems?", "",
              f"Dialogs where the tutor set a new practice problem: N {arms['N']['posed_new_problem']}, "
              f"H {arms['H']['posed_new_problem']}, T {arms['T']['posed_new_problem']} (pattern match on tutor turns).", "",
              "| T dialogs | M: T - N | U: T - N |", "|---|---|---|"]
    for name, c in split.items():
        n = c["M"]["n"] if c["M"] else 0
        lines.append(f"| {name} (n={n}) | {fmt(c['M'])} | {fmt(c['U'])} |")
    view_lines, view_data = views_section(run, posed["T"], rng, pilot_m={a: level[a]["M"] for a in ARMS})
    lines += view_lines
    # Its own RNG, so adding S and W leaves every other interval as it was.
    extra_lines, extra_data = extra_arms_section(run, level, arms, np.random.default_rng(1))
    lines += extra_lines
    # P2 and P3 are scored on every test view: U and M from this run's own
    # passes, then the views.py views (whose M, when present, replaces ours).
    tests = {v: {"T-N": contrasts["T-N"][v], "H-N": contrasts["H-N"][v]} for v in ("U", "M")}
    for v, d in (view_data or {}).get("views", {}).items():
        tests[v] = {"T-N": d["T-N"], "H-N": d["H-N"]}
    p4 = (view_data or {}).get("learner", {}).get("P4", {}).get("T-N")
    score_lines, score_data = scorecard(contrasts["T-N"]["P"], tests, p4)
    lines += score_lines
    lines += ["", "## Dialog health", "", "| Arm | Dialogs | Tutor turns | Handover | Blanked turns | Failed student calls | Unboxed answers | End reasons |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in present:
        a = arms[arm]
        lines.append(f"| {arm} | {a['dialogs']} | {a['tutor_turns']:.2f} | {a['handover_rate']:.0%} | {a['blanked_turns']:.1%}"
                     f" | {a['failed_student_calls']} of {a['student_calls']} | {a['unboxed_answers']} | {a['end_reasons']} |")
    if cost:
        total = sum(c["cost_usd"] for c in cost.values())
        lines += ["", f"## Cost: ${total:.2f}", "", "| Model | Calls | Input tokens | Output tokens | USD |", "|---|---|---|---|---|"]
        for model, c in sorted(cost.items()):
            lines.append(f"| {model} | {c['calls']:,} | {c['prompt_tokens']:,} | {c['completion_tokens']:,} | {c['cost_usd']:.2f} |")
        if projection:
            lines += ["", f"Projected 300-problem audit: {projection['audit_gpu_hours']:.1f} A100-hours at "
                      f"{GPU_TOKENS_PER_S:,} tokens/s, or about ${projection['audit_api_usd']:.0f} of student calls on OpenRouter."]
    providers = sorted({(s["model"], s["provider"]) for row in ledger for s in row.get("served_by", [])})
    if providers:
        lines += ["", "Served by: " + "; ".join(f"{m} via {p}" for m, p in providers)]
    reasoned = {m: n for row in ledger for m, n in row.get("reasoned", {}).items() if n and m == JUDGE}
    if reasoned:
        lines += [f"WARNING: the judge returned reasoning text on {sum(reasoned.values())} calls; check its settings."]

    report = "\n".join(lines) + "\n"
    out = Path(args.report) if args.report else run / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"design": design, "gates": gates, "decision": decision, "framing": framing, "arms": arms, "posed_split": split,
         "contrasts": contrasts, "pretest": pretest, "cost": cost, "projection": projection,
         "views": view_data, "extra_arms": extra_data, "scorecard": score_data}, indent=2))
    print(report)
    print(f"Written to {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
