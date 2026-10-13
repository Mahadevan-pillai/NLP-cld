"""Evaluation figures (shares the EDA figure style)."""

from pathlib import Path

import numpy as np

from src.data.eda_plots import INK, INK_2, SURFACE, _finish, plt

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # blue ramp (palette)


def confusion_matrix_figure(tn: int, fp: int, fn: int, tp: int, names: dict, title: str, path: Path) -> Path:
    """Counts + row percentages. Colour = row % on a FIXED 0-100 scale (same for every figure)."""
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.array([[tn, fp], [fn, tp]])
    row_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100)
    labels = [names[0].replace("_", "-"), names[1].replace("_", "-")]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)", ha="center", va="center", fontsize=10,
                    color=SURFACE if row_pct[i, j] > 55 else INK)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return _finish(fig, [], path, note=f"Validation set (n = {cm.sum():,}). % = share of the true class (row).")
