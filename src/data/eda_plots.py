"""Publication-quality EDA figures (matplotlib, static PNG, 300 dpi).

Style: two validated categorical colours (CVD-safe pair, checked with
the dataviz palette validator) for the two classes, a single colour for
one-series charts, text in neutral ink (never the series colour), a
light horizontal/vertical grid only, no top/right spines.

Term charts show LATIN-SCRIPT terms only: matplotlib cannot shape
Devanagari (conjuncts and vowel signs would render incorrectly).
Devanagari terms are fully reported in the CSV tables instead.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}   # palette slots 1 (blue) and 2 (orange)
SINGLE = "#2a78d6"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dd", "#ffffff"
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": INK_2, "axes.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def _finish(fig, ax_list, path: Path, grid_axis: str = "y", note: str | None = None) -> Path:
    for ax in ax_list:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    bottom = 0
    if note:
        wrapped = textwrap.fill(note, width=int(fig.get_figwidth() * 17))
        lines = len(wrapped.splitlines())
        fig.text(0.01, 0.01, wrapped, fontsize=7.5, color=INK_2, ha="left", va="bottom")
        bottom = (0.16 * lines) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return path


def class_distribution(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    labels = [n.replace("_", "-") for n in table["label_name"]]
    bars = ax.bar(labels, table["count"], width=0.55, linewidth=0,
                  color=[CLASS_COLORS[c] for c in table["target_label"]])
    for bar, count, pct in zip(bars, table["count"], table["percent"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, table["count"].max() * 1.22)
    ax.set_ylabel("Samples")
    ax.set_title(f"Primary label distribution (n = {table['count'].sum():,})")
    return _finish(fig, [ax], path)


def histogram(values: pd.Series, path: Path, title: str, xlabel: str, bins=40, note=None) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(values, bins=bins, color=SINGLE, edgecolor=SURFACE, linewidth=0.6)
    med = float(pd.Series(values).median())
    ax.axvline(med, color=INK_2, linewidth=1.2, linestyle="--")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.08)
    ax.annotate(f"median {med:.3g}", xy=(med, ymax * 1.04), xytext=(6, 0), textcoords="offset points",
                color=INK_2, fontsize=8, va="center", ha="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")
    ax.set_title(title)
    return _finish(fig, [ax], path, note=note)


def length_by_class(features: pd.DataFrame, names: dict, path: Path) -> Path:
    present = features[features["has_text"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, col, label in zip(axes, ["char_count", "word_count"], ["Characters", "Words (whitespace-separated)"]):
        data = [present.loc[present["target_label"] == k, col] for k in names]
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                         medianprops={"color": INK, "linewidth": 1.5},
                         flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4,
                                     "markeredgecolor": INK_2, "markerfacecolor": "none"})
        for patch, k in zip(box["boxes"], names):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.85)
            patch.set_edgecolor(INK_2)
        ax.set_xticks(range(1, len(names) + 1),
                      [f"{n.replace('_', '-')}\n(n = {len(d):,})" for n, d in zip(names.values(), data)])
        ax.set_ylabel(label)
    axes[0].set_title("Text length by class")
    axes[1].set_title("Word count by class")
    return _finish(fig, axes, path, note="Non-empty texts only. Boxes: IQR; line: median; whiskers: 1.5 IQR; "
                                         "circles: individual samples beyond the whiskers.")


def top_terms(table: pd.DataFrame, path: Path, title: str, top: int = 20) -> Path:
    latin = table[table["script"] == "latin"].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.barh(latin["term"], latin["count"], color=SINGLE, edgecolor=SURFACE, linewidth=1.5, height=0.7)
    for y, c in enumerate(latin["count"]):
        ax.text(c, y, f" {c:,}", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Occurrences")
    ax.set_title(title)
    return _finish(fig, [ax], path, grid_axis="x",
                   note="Latin-script terms only (Devanagari terms are in token_frequency.csv). "
                        "EDA-only tokenization: lowercase, letter runs; no stopword removal.")


def distinctive_terms(table: pd.DataFrame, names: dict, path: Path, top: int = 15) -> Path:
    latin = table[table["script"] == "latin"]
    pos = latin[latin["z_score"] > 0].head(top)
    neg = latin[latin["z_score"] < 0].head(top)
    # bottom -> top: most negative ... most positive
    data = pd.concat([neg.sort_values("z_score"), pos.sort_values("z_score")])
    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    colors = [CLASS_COLORS[1] if z > 0 else CLASS_COLORS[0] for z in data["z_score"]]
    ax.barh(data["term"], data["z_score"], color=colors, edgecolor=SURFACE, linewidth=1.2, height=0.75)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Weighted log-odds z-score (descriptive, not a significance test)")
    ax.set_title("Terms most associated with each class")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CLASS_COLORS[1], label=f"{names[1].replace('_', '-')} (z > 0)"),
                       Patch(color=CLASS_COLORS[0], label=f"{names[0].replace('_', '-')} (z < 0)")],
              loc="lower right", fontsize=8)
    return _finish(fig, [ax], path, grid_axis="x",
                   note="Latin-script terms with >= 10 occurrences. Log-odds with informative Dirichlet prior.")


def grouped_percent_by_class(table: pd.DataFrame, category_col: str, names: dict, path: Path,
                             title: str, xlabel: str, note: str | None = None, label_map=None) -> Path:
    cats = list(table[category_col])
    nice = [(label_map or {}).get(c, c).replace("_", " ") for c in cats]
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(cats) + 1.4))
    height = 0.38
    ys = range(len(cats))
    for i, (k, name) in enumerate(names.items()):
        vals = table[f"percent_{name}"]
        pos = [y + (i - 0.5) * height for y in ys]
        ax.barh(pos, vals, height=height, color=CLASS_COLORS[k], edgecolor=SURFACE, linewidth=1.2,
                label=name.replace("_", "-"))
        for p, v in zip(pos, vals):
            ax.text(v, p, f" {v:.1f}%", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(list(ys), nice)
    ax.invert_yaxis()
    ax.set_xlim(0, max(table[[f"percent_{n}" for n in names.values()]].max()) * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return _finish(fig, [ax], path, grid_axis="x", note=note)


def handles_by_class(table: pd.DataFrame, names: dict, path: Path) -> Path:
    data = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(data) + 1.6))
    left = pd.Series(0, index=data.index)
    for k, name in names.items():
        ax.barh(data["handle"], data[f"count_{name}"], left=left, color=CLASS_COLORS[k], edgecolor=SURFACE,
                linewidth=1.5, height=0.6, label=name.replace("_", "-"))
        left = left + data[f"count_{name}"]
    for y, total in enumerate(data["samples"]):
        ax.text(total, y, f" {total}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Samples containing the @handle")
    ax.set_title("@handles in meme text, by class")
    ax.legend(loc="lower right", fontsize=8)
    return _finish(fig, [ax], path, grid_axis="x",
                   note="Handles in >= 5 samples. Likely page watermarks captured by OCR: a potential shortcut feature.")


def secondary_combinations(table: pd.DataFrame, path: Path, n_positive: int) -> Path:
    data = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.barh(data["combination"].str.replace("+", " + ", regex=False), data["count"], color=SINGLE,
            edgecolor=SURFACE, linewidth=1.5, height=0.65)
    for y, c in enumerate(data["count"]):
        ax.text(c, y, f" {c:,} ({100 * c / n_positive:.1f}%)", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, data["count"].max() * 1.35)
    ax.set_xlabel("Misogynistic samples")
    ax.set_title(f"Sub-label combinations (n = {n_positive:,} misogynistic)")
    return _finish(fig, [ax], path, grid_axis="x")
