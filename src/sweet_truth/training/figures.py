"""Confusion-matrix figures for the model comparison.

Each panel follows the classic layout: the 2x2 grid of true / false positives and negatives,
sensitivity and specificity beside the rows, precision, negative predictive value and accuracy
below the columns. Sweeteners are the positive class.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Violet (the app's accent) marks correct predictions, amber marks errors. The two stay
# clearly apart for every kind of colour blindness (OKLab difference 29-36), and every
# cell is also labelled in words, so colour is never the only cue.
PALETTE = {
    "correct_fill": "#ece9fb",
    "correct_edge": "#6250d6",
    "error_fill": "#fdf0dc",
    "error_edge": "#d97706",
    "measure_fill": "#f4f3f0",
    "measure_edge": "#d6d4ce",
    "ink": "#1f1d2b",
    "ink_muted": "#5c5a66",
    "background": "#ffffff",
}

SET_TITLES = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test (hold-out lists)",
    "unseen": "Unseen ingredients",
    "challenge": "Challenge set",
}


def _cell(ax: Axes, x: float, y: float, fill: str, edge: str, lines: list[tuple[str, dict]]):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.04, y + 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4,
        )
    )
    step = 0.2
    top = y + 0.5 + step * (len(lines) - 1) / 2
    for i, (text, style) in enumerate(lines):
        ax.text(x + 0.5, top - i * step, text, ha="center", va="center", **style)


def confusion_panel(ax: Axes, m: Mapping[str, Any], title: str) -> None:
    """Draw one confusion matrix with its measures, from `binary_metrics` output."""
    ink, muted = PALETTE["ink"], PALETTE["ink_muted"]
    label = {"fontsize": 7.5, "color": muted}
    count = {"fontsize": 14, "color": ink, "fontweight": "bold"}
    tag = {"fontsize": 7, "color": "#9a4f00", "fontweight": "bold"}
    measure = {"fontsize": 12.5, "color": ink, "fontweight": "bold"}
    n = max(m["n"], 1)

    def share(value: int) -> tuple[str, dict]:
        return f"{value / n:.0%} of {m['n']}", {"fontsize": 7, "color": muted}

    # Grid: columns 0-1 are predicted sweetener / other, column 2 holds row measures.
    # Rows 2-1 (top to bottom) are actual sweetener / other, row 0 holds column measures.
    correct = (PALETTE["correct_fill"], PALETTE["correct_edge"])
    error = (PALETTE["error_fill"], PALETTE["error_edge"])
    grey = (PALETTE["measure_fill"], PALETTE["measure_edge"])
    _cell(ax, 0, 2, *correct, [("True positive", label), (str(m["tp"]), count), share(m["tp"])])
    _cell(
        ax,
        1,
        2,
        *error,
        [("False negative", label), (str(m["fn"]), count), ("Type II error", tag)],
    )
    _cell(
        ax,
        0,
        1,
        *error,
        [("False positive", label), (str(m["fp"]), count), ("Type I error", tag)],
    )
    _cell(ax, 1, 1, *correct, [("True negative", label), (str(m["tn"]), count), share(m["tn"])])
    _cell(
        ax,
        2,
        2,
        *grey,
        [("Sensitivity", label), ("(recall)", label), (f"{m['recall']:.2f}", measure)],
    )
    _cell(ax, 2, 1, *grey, [("Specificity", label), (f"{m['specificity']:.2f}", measure)])
    _cell(ax, 0, 0, *grey, [("Precision", label), (f"{m['precision']:.2f}", measure)])
    _cell(
        ax,
        1,
        0,
        *grey,
        [("Negative", label), ("predictive value", label), (f"{m['npv']:.2f}", measure)],
    )
    _cell(ax, 2, 0, *grey, [("Accuracy", label), (f"{m['accuracy']:.2f}", measure)])

    head = {"fontsize": 8.5, "color": ink, "fontweight": "bold"}
    ax.text(0.5, 3.12, "Sweetener", ha="center", va="bottom", **head)
    ax.text(1.5, 3.12, "Other", ha="center", va="bottom", **head)
    ax.text(1.0, 3.38, "Predicted", ha="center", va="bottom", fontsize=8, color=muted)
    ax.text(-0.1, 2.5, "Sweetener", ha="right", va="center", rotation=90, **head)
    ax.text(-0.1, 1.5, "Other", ha="right", va="center", rotation=90, **head)
    ax.text(-0.42, 2.0, "Actual", ha="right", va="center", rotation=90, fontsize=8, color=muted)
    ax.set_title(f"{title}\nF1 {m['f1']:.2f}", fontsize=10, color=ink, pad=2, loc="center")
    ax.set_xlim(-0.6, 3.05)
    ax.set_ylim(-0.05, 3.62)
    ax.set_aspect("equal")
    ax.axis("off")


def _figure(rows: int, cols: int) -> tuple[plt.Figure, list[list[Axes]]]:
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 4.4 * rows), squeeze=False, facecolor=PALETTE["background"]
    )
    return fig, axes


def _short(name: str) -> str:
    return name.split(" (")[0]


def draw_all(reports: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    """The figures used in the README, the reports and the evaluation notebook.

    - confusion_v2.png: the app's model on seen (test) and never-seen ingredients
    - confusion_overview.png: v1 and v2 on every set
    - confusion_unseen.png, confusion_challenge.png: every model on the two sets that
      separate them
    """
    output.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig: plt.Figure, name: str, dpi: int = 160) -> None:
        path = output / name
        fig.savefig(path, dpi=dpi, facecolor=PALETTE["background"])
        plt.close(fig)
        written.append(path)

    by_name = {_short(r["model"]): r for r in reports}
    v2 = by_name.get("v2")
    if v2:
        fig, axes = _figure(1, 2)
        confusion_panel(axes[0][0], v2["sets"]["test"], "Ingredients seen in training (test)")
        confusion_panel(axes[0][1], v2["sets"]["unseen"], "Ingredients never seen in training")
        fig.suptitle("Model v2", fontsize=13, color=PALETTE["ink"], fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "confusion_v2.png")

    for key in ("unseen", "challenge"):
        scored = [r for r in reports if r["sets"].get(key)]
        fig, axes = _figure(1, len(scored))
        for ax, report in zip(axes[0], scored, strict=True):
            confusion_panel(ax, report["sets"][key], report["model"])
        fig.suptitle(SET_TITLES[key], fontsize=13, color=PALETTE["ink"], fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, f"confusion_{key}.png")

    main = [by_name[n] for n in ("v1", "v2") if n in by_name]
    keys = [k for k in SET_TITLES if all(r["sets"].get(k) for r in main)]
    fig, axes = _figure(len(keys), len(main))
    for row, key in zip(axes, keys, strict=True):
        for ax, report in zip(row, main, strict=True):
            confusion_panel(
                ax, report["sets"][key], f"{SET_TITLES[key]} · {_short(report['model'])}"
            )
    fig.tight_layout()
    save(fig, "confusion_overview.png", dpi=130)
    return written
