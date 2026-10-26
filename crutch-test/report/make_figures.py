"""Make the Part 2 report's figures from the pilot results.

Usage (from the repo root, any Python with matplotlib):
    python crutch-test/report/make_figures.py [--extra-l crutch-test/results/pilot50v2_lr0.03.json]

Writes figures/effects.{pdf,png} and figures/workflow.{pdf,png} next to this file.
--extra-l adds a second L row from another analyze.py report (the lr 0.03 rerun).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "figures"

# Ink and chrome from the dataviz reference palette; the two hues passed its
# validator (CVD and normal-vision separation, contrast) on a light surface.
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FROZEN, LEARNING = "#2a78d6", "#eb6834"

# Bastani et al. (PNAS 2025), Table 1: unaided exam effects as absolute
# differences in proportion correct, with SEs; 95% CI = estimate +/- 1.96 SE.
HUMAN = {"T-N": (-0.054, 0.022), "H-N": (-0.004, 0.013)}
LABELS = {
    "M": "M: own turns, tutor hidden",
    "C": "C: M minus copied text",
    "K": "K: one note (≤120 tokens)",
    "L": "L: trained on own turns",
}
P_MARGIN = 0.03  # the pass rules' fixed margin (P2 and P3)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable
})


def rows(report: dict, extra: dict | None, extra_label: str) -> list[dict]:
    """One row per simulator, top to bottom: the human trial first, then M, C, K, L."""
    out = [{"label": "Human trial (Bastani et al.)", "color": INK, "marker": "D",
            **{c: {"mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se} for c, (m, se) in HUMAN.items()}}]
    views = report["views"]["views"]
    for v in ("M", "C", "K", "L"):
        label = LABELS[v] + (f" (lr {report['views']['learner']['lr']:g})" if v == "L" else "")
        out.append({"label": label, "color": LEARNING if v == "L" else FROZEN, "marker": "o",
                    "T-N": views[v]["T-N"], "H-N": views[v]["H-N"]})
    if extra is not None:
        ev = extra["views"]["views"]["L"]
        out.append({"label": f"{LABELS['L']} ({extra_label})", "color": LEARNING, "marker": "s",
                    "T-N": ev["T-N"], "H-N": ev["H-N"]})
    return out


def effects(data: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 0.42 * len(data) + 1.0), sharey=True)
    ys = list(range(len(data)))[::-1]
    # The full gates (G2 = analyze.py's P2, G3 = P3) go in the caption; the dashed line marks the margin.
    panels = (("T-N", "(a) Answers vs. no AI: T $-$ N", "$\\mathcal{G}_2$ margin"),
              ("H-N", "(b) Hints vs. no AI: H $-$ N", "$\\mathcal{G}_3$ margin"))
    for ax, (key, title, rule) in zip(axes, panels):
        ax.axvline(0, color=BASE, lw=1, zorder=1)
        ax.axvline(-P_MARGIN, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(-P_MARGIN - 0.006, -0.5, rule, ha="right", va="bottom", fontsize=6.5, color=MUTED)
        for y, r in zip(ys, data):
            e = r[key]
            ax.plot([e["lo"], e["hi"]], [y, y], color=r["color"], lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot(e["mean"], y, marker=r["marker"], ms=6, color=r["color"], mec="white", mew=0.8, zorder=3)
            ax.text(e["hi"] + 0.012, y, f"{e['mean']:+.3f}", va="center", fontsize=6.5, color=INK2)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.14, 0.33)
        ax.set_ylim(-0.6, len(data) - 0.1)
    axes[0].set_yticks(ys, [r["label"] for r in data])
    for ax in axes:  # rows are labelled on the left panel; no tick marks on either
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Unaided post-test difference vs. no AI (proportion correct, 95% CI)", fontsize=7.5, color=INK2)
    handles = [Line2D([], [], color=c, marker=m, lw=1.8, ms=6, mec="white", label=lab) for c, m, lab in
               ((INK, "D", "Human trial"), (FROZEN, "o", "Frozen simulators"), (LEARNING, "o", "Learning student"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 30)
    ax.axis("off")
    # (x, width, title, [(line, color), ...]); widths sum with 3-unit gaps to 100.
    boxes = [
        (0, 12, "Problem", [("Big-Math", INK2), ("problem +", INK2), ("unseen variant", INK2)]),
        (15, 24, "Tutoring dialog\n(≤ 8 tutor turns)", [
            ("N: no AI (\"Keep going\")", INK2), ("H: hints (GPT Tutor prompt)", INK2),
            ("T: answers (GPT Base prompt)", INK2)]),
        (42, 22, "What the student\ncarries into the test", [
            ("M: own turns, tutor hidden", FROZEN), ("C: M minus copied text", FROZEN),
            ("K: one note (≤120 tokens)", FROZEN), ("L: LoRA on own tokens", LEARNING)]),
        (67, 13, "Unaided\npost-test", [("on the variant", INK2), ("16 samples", INK2)]),
        (83, 17, "Score vs.\nhuman trial", [("paired by problem", INK2), ("vs. N, bootstrap CI", INK2),
                                             ("gates $\\mathcal{G}_1$–$\\mathcal{G}_4$", INK2)]),
    ]
    for x, w, title, lines in boxes:
        ax.add_patch(FancyBboxPatch((x + 0.4, 1), w - 0.8, 27, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="white", ec=BASE, lw=1))
        ax.text(x + w / 2, 26.5, title, ha="center", va="top", fontsize=7, color=INK, weight="bold",
                linespacing=1.1)
        for i, (line, color) in enumerate(lines):
            ax.text(x + w / 2, 17.5 - 4 * i, line, ha="center", va="top", fontsize=6, color=color)
    for (x, w, *_), (nx, *_) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(nx + 0.2, 14.5), xytext=(x + w - 0.2, 14.5),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=0))
    ax.text(50, -1.5, "Student: Llama-3.1-8B-Instruct (bf16)  ·  Tutor: Claude Haiku 5.5  ·  "
            "Handover judge: Qwen3.6-27B", ha="center", va="top", fontsize=6, color=MUTED)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--report", default=str(RESULTS / "pilot50v2.json"))
    ap.add_argument("--extra-l", default=None, help="another analyze.py report whose L row to add")
    ap.add_argument("--extra-label", default="lr 0.03")
    args = ap.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    extra = json.loads(Path(args.extra_l).read_text(encoding="utf-8")) if args.extra_l else None
    OUT.mkdir(parents=True, exist_ok=True)
    effects(rows(report, extra, args.extra_label), OUT / "effects")
    workflow(OUT / "workflow")
    print(f"wrote effects and workflow figures to {OUT}")


if __name__ == "__main__":
    main()
