from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import DIAGNOSIS_COLORS as COLORS
from .config import FIGURE, THERAPEUTIC_COLORS
from .ml.ajcc import grouped_ajcc_labels

# --- Shared manuscript style -------------------------------------------------
# Every figure is drawn at the width of the panel slot it is destined for (see :func:`panel_size`) rather than at an
# arbitrary inch size, so assembling panels in panels.py never needs a rescale. R/lib/plot_utils.R mirrors these
# settings from the same config block. Figures are therefore small: a 'third' panel is 56 mm wide.
_MM_PER_IN = 25.4

BASE_SIZE: float = FIGURE['base_size']
SMALL_SIZE: float = FIGURE['small_size']   # dense in-plot annotations; the guideline floor, never go below

# Vertical space a faceted boxplot spends on everything that is not plotting area, measured off rendered figures.
# Per facet row: its title plus a rotated set of group labels. Per figure: the two-line shared x axis label.
FACET_ROW_CHROME_MM = 14.0
FIGURE_CHROME_MM = 8.0

# Embedding scatters size their box to the point cloud (see :func:`plot_embedding`). The floor stops a very flat
# embedding from collapsing to a strip.
EMBEDDING_ASPECT_RANGE = (0.25, 1.0)
EMBEDDING_LEGEND_MM = 6.0   # band above the axes for the one-row legend, measured off rendered figures

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': list(FIGURE['font_family']),
    'font.size': BASE_SIZE,
    'axes.titlesize': BASE_SIZE,
    'axes.labelsize': BASE_SIZE,
    'xtick.labelsize': BASE_SIZE,
    'ytick.labelsize': BASE_SIZE,
    'legend.fontsize': BASE_SIZE,
    'figure.titlesize': BASE_SIZE,
    'axes.linewidth': FIGURE['line_width'],
    'grid.linewidth': FIGURE['line_width'],
    'lines.linewidth': FIGURE['line_width'],
    'patch.linewidth': FIGURE['line_width'],
    'xtick.major.width': FIGURE['line_width'],
    'ytick.major.width': FIGURE['line_width'],
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,   # embed TrueType rather than outlining glyphs, so text stays editable in Illustrator/Inkscape
    'savefig.dpi': 600,   # only bites for raster output; the PDFs stay vector
})


def panel_size(
    slot: str = 'half',
    aspect: float = 0.75,
    height_mm: Optional[float] = None,
) -> tuple[float, float]:
    """Figure size in inches for a panel destined for the ``slot`` column width of an assembled figure.

    :param slot: Key into the config's ``figure.panel_mm`` (``full``/``two_thirds``/``half``/``third``).
    :param aspect: Height as a fraction of the width; ignored when ``height_mm`` is given.
    :param height_mm: Explicit height in mm, for panels whose height grows with the number of rows plotted.
        Clamped to the guideline maximum (``figure.max_height_mm``, a full page).
    :return: ``(width_in, height_in)`` for ``plt.subplots(figsize=...)``.
    """
    width_mm = FIGURE['panel_mm'][slot]
    height_mm = min(height_mm if height_mm is not None else width_mm * aspect, FIGURE['max_height_mm'])
    return width_mm / _MM_PER_IN, height_mm / _MM_PER_IN


def plot_roc_curves(
    fpr: dict[str, np.ndarray],
    tpr: dict[str, np.ndarray],
    roc_auc: dict[str, float],
    title: Optional[str] = None,
    slot: str = 'third',
    out_path: Optional[Path] = None,
) -> None:
    """
    Plots per-class and macro-average ROC curves.

    :param fpr: Dict keyed by label name mapping to arrays of false positive rates.
    :param tpr: Dict keyed by label name mapping to arrays of true positive rates.
    :param roc_auc: Dict keyed by label name mapping to AUC values.
    :param title: Panel title; omitted when ``None``. Pass a model identifier when several of these end up side by
        side in one panel figure.
    :param slot: Panel width slot (see :func:`panel_size`); drawn square, as ROC axes share a 0-1 range.
    :param out_path: If given, save the figure to this path; otherwise call ``plt.show()``.
    :return: None
    """
    plt.figure(figsize=panel_size(slot, aspect=1.0))
    for label in fpr:
        style = '--' if label == 'macro' else '-'
        color = COLORS.get(label, 'black')
        plt.plot(fpr[label], tpr[label], style, color=color,
                 label=f'{label} (AUC = {roc_auc[label]:.3f})')
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if title:
        plt.title(title)
    plt.legend(loc='lower right', fontsize=SMALL_SIZE)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: Optional[str] = None,
    slot: str = 'third',
    out_path: Optional[Path] = None,
) -> None:
    """
    Plots a confusion matrix with annotated counts.

    :param cm: Pre-computed confusion matrix as a numpy array of shape (n_classes, n_classes).
    :param class_names: List of class name strings used for axis labels.
    :param title: Panel title; omitted when ``None``. Pass a model identifier when several of these are composited
        side by side, as the matrices are otherwise indistinguishable.
    :param slot: Panel width slot (see :func:`panel_size`).
    :param out_path: If given, save the figure to this path; otherwise call ``plt.show()``.
    :return: None
    """
    # Row-normalize (per true class) for coloring and percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=panel_size(slot, aspect=0.95))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_norm[i, j] > 0.5 else '#052360'
            ax.text(j, i, f'{cm[i, j]}\n{cm_norm[i, j]:.1%}', ha='center', va='center', color=color,
                    fontsize=SMALL_SIZE)

    # Colorbar with 0-100% ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{int(t * 100)}%' for t in np.linspace(0, 1, 6)])
    cbar.outline.set_linewidth(FIGURE['line_width'])

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def plot_per_group_mae(
    per_group_mae: pd.Series,
    ci: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    slot: str = 'third',
    out_path: Optional[Path] = None,
) -> None:
    """
    Plots mean absolute error per therapeutic group as a vertical bar chart.

    :param per_group_mae: Series of MAE values indexed by therapeutic group code. Missing groups are skipped; the
        codes are shown as their combined AJCC-stage labels (see :func:`ml.ajcc.grouped_ajcc_labels`).
    :param ci: Optional frame with ``ci_low``/``ci_high`` columns indexed like ``per_group_mae``; drawn as
        (possibly asymmetric) error bars. Groups without a CI get no bar.
    :param title: Panel title; omitted when ``None``. Pass a model identifier when composited alongside other models.
    :param slot: Panel width slot (see :func:`panel_size`).
    :param out_path: If given, save the figure to this path; otherwise call ``plt.show()``.
    :return: None
    """
    per_group_mae = per_group_mae.dropna().sort_index()

    yerr = None
    if ci is not None:
        ci = ci.reindex(per_group_mae.index)
        lower = (per_group_mae - ci['ci_low']).clip(lower=0).to_numpy()
        upper = (ci['ci_high'] - per_group_mae).clip(lower=0).to_numpy()
        yerr = np.vstack([lower, upper])

    # Colors are keyed by group code, so keep the index numeric and only the tick labels show the AJCC stages.
    group_labels = grouped_ajcc_labels()
    tick_labels = [group_labels.get(g, str(g)) for g in per_group_mae.index]
    bar_colors = [THERAPEUTIC_COLORS.get(g, 'steelblue') for g in per_group_mae.index]
    fig, ax = plt.subplots(figsize=panel_size(slot, aspect=0.85))
    ax.bar(tick_labels, per_group_mae.values,
           color=bar_colors, alpha=0.85, yerr=yerr, capsize=2,
           error_kw=dict(ecolor='black', alpha=0.6, lw=FIGURE['line_width']))
    ax.axhline(float(per_group_mae.mean()), color='black', lw=FIGURE['line_width'], ls='--',
               label=f'macro mean = {per_group_mae.mean():.2f}')
    ax.set_xlabel('Therapeutic group')
    ax.set_ylabel('Mean absolute error')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize=SMALL_SIZE)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    xlabel: str = 'Permutation importance (drop in AUROC)',
    title: Optional[str] = None,
    slot: str = 'half',
    out_path: Optional[Path] = None,
) -> None:
    """
    Plots the top-N features by mean permutation importance as a horizontal bar chart with std error bars.

    :param importance_df: DataFrame with columns ``feature``, ``importance_mean``, ``importance_std``,
        sorted by ``importance_mean`` descending.
    :param top_n: Number of top features to plot (default: 30).
    :param xlabel: Label for the importance axis (default mentions AUROC; pass a different string when
        the underlying score is something else, e.g. balanced accuracy).
    :param title: Panel title; omitted when ``None``. Pass a model identifier when composited alongside other models.
    :param slot: Panel width slot (see :func:`panel_size`). Height grows with the number of bars, so a long feature
        list yields a tall panel rather than compressed rows. Half width by default: the y axis is a column of
        feature names, which at third width leaves too little room for the bars.
    :param out_path: If given, save the figure to this path; otherwise call ``plt.show()``.
    :return: None
    """
    top = importance_df.head(top_n).iloc[::-1]  # reverse so largest is at top of plot

    # 3.5 mm per bar keeps the tick labels from colliding at base_size; the rest is axis/title chrome.
    fig, ax = plt.subplots(figsize=panel_size(slot, height_mm=20 + 3.5 * len(top)))
    ax.barh(top['feature'], top['importance_mean'], xerr=top['importance_std'],
            color='steelblue', alpha=0.85,
            error_kw=dict(ecolor='black', alpha=0.5, lw=FIGURE['line_width']))
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.axvline(0, color='black', lw=FIGURE['line_width'])
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def plot_embedding(
    coords: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    xlabel: str = 't-SNE 1',
    ylabel: str = 't-SNE 2',
    title: Optional[str] = None,
    slot: str = 'half',
    aspect: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> None:
    """
    Plots a two-dimensional sample embedding as a scatter, coloured by group.

    Train/test split (when both are present) is encoded by marker shape.

    The panel's height follows the *embedding's own* aspect ratio (clamped by :data:`EMBEDDING_ASPECT_RANGE`). Pass
    ``aspect`` to override that for panels that have to line up with a sibling — two embeddings of different clouds
    otherwise come out at different heights.

    :param coords: Embedding indexed by sample, with columns ``dim1``, ``dim2``, ``group`` and ``split``.
    :param order: Group labels in display (and legend) order; labels absent from ``coords`` are skipped.
    :param colors: Group label -> colour; groups missing here fall back to grey.
    :param xlabel: Label for the first embedding dimension.
    :param ylabel: Label for the second embedding dimension.
    :param title: Panel title; omitted when ``None``.
    :param slot: Panel width slot (see :func:`panel_size`). Half width by default.
    :param aspect: Panel height as a fraction of its width, overriding the cloud's own aspect ratio.
    :param out_path: If given, save the figure to this path; otherwise call ``plt.show()``.
    :return: None
    """
    # Filled circles for the training cohort, open triangles for test.
    split_styles = {'train': ('o', True), 'test': ('^', False)}

    # Match the box to the point cloud, so equal scaling costs no blank panel. The floor keeps a very flat embedding
    # from becoming a strip too shallow for its own axis labels; the extra room above the axes is the legend's.
    spans = [coords[d].max() - coords[d].min() for d in ('dim1', 'dim2')]
    data_aspect = min(max(spans[1] / spans[0], min(EMBEDDING_ASPECT_RANGE)), max(EMBEDDING_ASPECT_RANGE))
    width_in, height_in = panel_size(slot, aspect=aspect if aspect is not None else data_aspect)
    fig, ax = plt.subplots(figsize=(width_in, height_in + EMBEDDING_LEGEND_MM / _MM_PER_IN))

    present = [g for g in order if g in coords['group'].values]
    for group in present:
        for split, (marker, filled) in split_styles.items():
            sub = coords[(coords['group'] == group) & (coords['split'] == split)]
            if sub.empty:
                continue
            color = colors.get(group, 'grey')
            ax.scatter(sub['dim1'], sub['dim2'], marker=marker, s=6,
                       color=color if filled else 'none', edgecolors=color,
                       linewidths=FIGURE['line_width'], alpha=0.8 if filled else 1.0)

    # Both dimensions come out of the same embedding, so a unit of dim1 must read as a unit of dim2.
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('SW')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    # One horizontal legend above the axes.
    handles = [Line2D([], [], marker='o', ls='none', ms=2.5, color=colors.get(g, 'grey'), label=g) for g in present]
    handles += [Line2D([], [], marker=split_styles[s][0], ls='none', ms=2.5, color='black',
                       mfc='black' if split_styles[s][1] else 'none', label=s)
                for s in split_styles if s in coords['split'].values and coords['split'].nunique() > 1]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(handles),
              fontsize=SMALL_SIZE, columnspacing=1.0, handletextpad=0.3, borderpad=0)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def _draw_group_boxplot(
    ax,
    df: pd.DataFrame,
    marker_col: str,
    group_col: str,
    order: list[str],
    annotations: dict[str, str],
    facet_mm: float,
    colors: Optional[dict[str, str]] = None,
) -> None:
    """Draw one marker's group boxplot (jittered points + top-right annotation box) onto ``ax``.

    The x-axis groups follow ``order`` (only labels present in ``df`` are drawn); ``annotations`` maps
    ``label -> value string`` for the summary box. Axis labels/title are left to the caller, and so is the per-group
    ``n`` — it is the same in every panel (see :func:`plot_grouped_boxplots`).

    ``facet_mm`` is this axes' width on the page, used to decide whether the group labels still fit horizontally.
    """
    present = [s for s in order if s in df[group_col].values]
    positions = list(range(len(present)))

    group_data = [df.loc[df[group_col] == s, marker_col].dropna() for s in present]
    bp = ax.boxplot(group_data, positions=positions, widths=0.5, patch_artist=True,
                    flierprops=dict(ms=2, mec='black', mew=FIGURE['line_width']),
                    medianprops=dict(color='black', lw=FIGURE['line_width']))
    for patch, s in zip(bp['boxes'], present):
        patch.set_facecolor((colors or {}).get(s, 'lightblue'))
        patch.set_alpha(0.7)

    for i, (vals, s) in enumerate(zip(group_data, present)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.4, s=3, linewidths=0,
                   color=(colors or {}).get(s, 'steelblue'), zorder=3)

    # Rotate only when the labels would collide: each group gets facet_mm / n_groups of width, and a character of
    # base_size text is roughly 0.6 * base_size points wide.
    char_mm = 0.6 * BASE_SIZE * _MM_PER_IN / 72
    slot_mm = facet_mm / max(len(present), 1)
    rotate = max((len(str(s)) for s in present), default=0) * char_mm > slot_mm
    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=45 if rotate else 0, ha='right' if rotate else 'center')

    if annotations:
        # At facet width there is no empty corner for the box, so make headroom above the data first.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))
        ax.text(0.98, 0.97, '\n'.join(f'{k} = {v}' for k, v in annotations.items()),
                transform=ax.transAxes, ha='right', va='top', fontsize=SMALL_SIZE,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))


def plot_grouped_boxplots(
    df: pd.DataFrame,
    marker_cols: list[str],
    group_col: str,
    order: list[str],
    xlabel: str,
    ylabel: str,
    annotations: dict[str, dict[str, str]],
    title: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
    out_path: Optional[Path] = None,
    ncols: int = 3,
    slot: str = 'full',
    facet_aspect: float = 1.0,
    names: Optional[dict[str, str]] = None,
) -> None:
    """Faceted boxplots — one subplot per marker in ``marker_cols`` — sharing a categorical grouping variable.

    The grouped counterpart to R's ``make_facet_boxplot``: all markers of one modality land in a single figure
    (e.g. every EpiSCORE cell type), each panel keeping its own free y-scale and summary annotation. Task-agnostic:
    the caller supplies the group ``order`` (therapeutic groups for the ordinal task, diagnosis classes for classification).

    :param df: Long-form frame holding every ``marker_cols`` column and ``group_col``.
    :param marker_cols: Marker columns to draw, one panel each (panel title = column name).
    :param group_col: Categorical column defining the x-axis groups.
    :param order: Group labels in display order; only those present in ``df`` are drawn.
    :param xlabel: Shared x-axis label (e.g. ``'Therapeutic group'`` or ``'Diagnosis'``).
    :param ylabel: Shared y-axis label (e.g. ``'Estimated fraction'``).
    :param annotations: Mapping ``marker_col -> {label: value string}`` for each panel's top-right text box.
    :param title: Figure suptitle; omitted when ``None``.
    :param colors: Optional ``group -> color`` mapping for the box faces; defaults to ``'lightblue'``.
    :param out_path: If given, save there; otherwise ``plt.show()``.
    :param ncols: Maximum number of panel columns; rows wrap as needed.
    :param slot: Width slot for the whole faceted figure (see :func:`panel_size`). Facets divide it, so a
        three-column ``'full'`` figure gives each facet roughly the width — and hence the text scale — of a
        standalone ``'third'`` panel.
    :param facet_aspect: Height of a facet's *plotting area* as a fraction of facet width. 1.0 gives square plotting
        areas; lower it when several rows of facets would make the figure too tall to share a page with other panels.
        Chrome is added on top rather than eating into it.
    :param names: Optional ``marker_col -> facet title`` mapping; columns not in it keep their column name.
    """
    n = len(marker_cols)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil division
    # Height is facets plus chrome, not a plain aspect ratio of the whole figure: chrome is roughly fixed per facet
    # row, so a ratio would starve a one-facet panel of plotting area while a nine-facet one barely notices.
    facet_mm = FIGURE['panel_mm'][slot] / ncols
    height_mm = nrows * (facet_mm * facet_aspect + FACET_ROW_CHROME_MM) + FIGURE_CHROME_MM
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=panel_size(slot, height_mm=height_mm))

    # Per-group n is identical in every facet (same samples, different marker column), so state it once in the shared
    # x label rather than under every group of every panel. Counts are listed in x-axis order.
    present = [s for s in order if s in df[group_col].values]
    counts = df[group_col].value_counts()
    xlabel = f"{xlabel}\n(n = {'/'.join(str(counts[s]) for s in present)})"

    for idx, col in enumerate(marker_cols):
        ax = axes[idx // ncols][idx % ncols]
        _draw_group_boxplot(ax, df, col, group_col, order, annotations.get(col, {}),
                            facet_mm=facet_mm, colors=colors)
        ax.set_title((names or {}).get(col, col))

    for idx in range(n, nrows * ncols):  # hide unused panels
        axes[idx // ncols][idx % ncols].axis('off')

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
    else:
        plt.show()
    plt.close(fig)


def plot_marker_heatmap(
    values: pd.Series,
    pvalues: pd.Series,
    value_label: str,
    title: Optional[str] = None,
    diverging: bool = True,
    slot: str = 'third',
    horizontal: bool = False,
    out_path: Optional[Path] = None,
) -> None:
    """Single-strip heatmap of a per-marker statistic, annotated with significance stars.

    :param values: Statistic per marker (signed Spearman ρ for the ordinal task), indexed by a display label.
    :param pvalues: Matching p-values (same index) driving the significance stars.
    :param value_label: Colorbar / tick label (e.g. ``'Spearman ρ'`` or ``'ε²'``).
    :param title: Panel title; omitted when ``None`` (``value_label`` already names the statistic).
    :param diverging: ``True`` uses a symmetric diverging map centred on 0 (signed statistics); ``False`` a
        sequential map from 0 (unsigned effect sizes).
    :param slot: Panel width slot (see :func:`panel_size`).
    :param horizontal: Lay the markers along the x axis instead of the y axis — a wide, short strip rather than a
        tall, narrow column. Needs a wide ``slot``, since the cells must stay wide enough for the value labels.
    :param out_path: If given, save there; otherwise ``plt.show()``.
    """
    n = len(values)
    if horizontal:
        # Fixed height: one row of cells plus room for the rotated marker labels beneath it.
        figsize = panel_size(slot, height_mm=34)
        matrix = values.values.reshape(1, -1)
    else:
        figsize = panel_size(slot, height_mm=15 + 5.0 * n)
        matrix = values.values.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if diverging:
        vmax = max(abs(values.min()), abs(values.max()), 0.3)
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    else:
        vmax = max(values.max(), 0.1)
        im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    # Colorbar stays on the right in both orientations: underneath, it collides with the rotated marker labels.
    fig.colorbar(im, ax=ax, label=value_label, shrink=0.8)

    if horizontal:
        ax.set_xticks(range(n))
        ax.set_xticklabels(values.index, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels([value_label])
    else:
        ax.set_yticks(range(n))
        ax.set_yticklabels(values.index)
        ax.set_xticks([0])
        ax.set_xticklabels([value_label])

    for i, (val, p) in enumerate(zip(values, pvalues)):
        stars = p_stars(p)
        # White label over the dark end of each colormap (diverging: both extremes; sequential: the low end).
        hot = abs(val) > vmax * 0.6 if diverging else val < vmax * 0.5
        color = 'white' if hot else 'black'
        # Two decimals when horizontal: the cells are only as wide as the panel divided by the marker count.
        label = f'{val:.2f}{stars}' if horizontal else f'{val:.3f}{stars}'
        x, y = (i, 0) if horizontal else (0, i)
        ax.text(x, y, label, ha='center', va='center', color=color, fontsize=SMALL_SIZE)

    if title:
        ax.set_title(title)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def plot_pairwise_mwu_heatmap(
    values: pd.DataFrame,
    pvalues: pd.DataFrame,
    title: Optional[str] = None,
    slot: str = 'two_thirds',
    out_path: Optional[Path] = None,
) -> None:
    """Heatmap of the pairwise Mann-Whitney U effect size, group pairs down y and markers along x.

    Cells hold the common-language effect size ``U / (n1 * n2)`` — the probability that a random sample from the
    pair's first group exceeds one from its second — rather than the raw U, which scales with ``n1 * n2`` and so is
    not comparable across pairs of unequal size. The map therefore diverges around 0.5 (no rank separation) and is
    signed, unlike the unsigned omnibus ε². Cells whose value is NaN (no post-hoc run, e.g. the omnibus test was not
    significant) are drawn in grey, which the diverging map's white midpoint would otherwise read as a value.

    :param values: Effect sizes with the group pairs as the index (labelled ``'<1st> vs <2nd>'``, matching the
        colorbar) and the markers as columns.
    :param pvalues: Matching p-values (same shape and labels) driving the significance stars.
    :param title: Panel title; omitted when ``None``.
    :param slot: Panel width slot (see :func:`panel_size`). Needs a wide slot: the cells must stay wide enough for
        the value labels, and the markers run along x.
    :param out_path: If given, save there; otherwise ``plt.show()``.
    """
    n_pairs, n_markers = values.shape
    # One row of cells per pair plus room for the rotated marker labels beneath them. Kept tight: with three pairs
    # this panel closes a full-page marker figure (see panels.py), which leaves it little vertical room.
    figsize = panel_size(slot, height_mm=22 + 4.0 * n_pairs)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Symmetric around 0.5 so both directions of separation read equally, with a floor so a null marker set does not
    # get its noise stretched over the full colormap.
    matrix = values.values.astype(float)
    spread = max(np.nanmax(np.abs(matrix - 0.5)), 0.15) if np.isfinite(matrix).any() else 0.15
    cmap = plt.get_cmap('RdBu_r').with_extremes(bad='0.9')
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect='auto', vmin=0.5 - spread, vmax=0.5 + spread)
    # Terse label: spelled out ('P(group 1 > group 2)') the rotated text is taller than this panel's axes. Which
    # groups '1st'/'2nd' are is on every row tick.
    fig.colorbar(im, ax=ax, label='P(1st > 2nd)', shrink=0.9)

    ax.set_xticks(range(n_markers))
    ax.set_xticklabels(values.columns, rotation=45, ha='right')
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(values.index)

    for i in range(n_pairs):
        for j in range(n_markers):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            # White label over the dark end of either extreme of the diverging map.
            color = 'white' if abs(val - 0.5) > spread * 0.6 else 'black'
            # Two decimals: the cells are only as wide as the panel divided by the marker count.
            ax.text(j, i, f'{val:.2f}{p_stars(pvalues.values[i, j])}', ha='center', va='center',
                    color=color, fontsize=SMALL_SIZE)

    if title:
        ax.set_title(title)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def p_stars(p: float) -> str:
    """Significance stars for a p-value (``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, else none).

    Public so the analysis modules annotate with the same thresholds the figures drawn here use (e.g.
    :func:`ml.correlation._fmt_p`).
    """
    if not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def plot_metric_diff_forest(
    labels: list[str],
    deltas: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    pvalues: np.ndarray,
    significant: np.ndarray,
    xlabel: str,
    title: Optional[str] = None,
    slot: str = 'half',
    out_path: Optional[Path] = None,
) -> None:
    """Forest plot of a paired metric difference (Δ) between model pairs, with a reference line at Δ = 0.

    One row per model pair: a marker at the point estimate and, when finite, a horizontal 95% CI whisker. Pairs whose
    corrected p-value clears the threshold are drawn in the emphasis colour. Rows with a non-finite CI (e.g. a
    degenerate bootstrap, or a test reporting only a p-value) are drawn as a bare point. Each row is annotated with its
    p-value and significance stars.

    :param labels: One label per model pair (row), labelled ``'<1st> vs <2nd>'`` to match the sign convention named in
        ``xlabel``, top-to-bottom in the given order.
    :param deltas: Point estimate of the metric difference per pair.
    :param ci_low: Lower 95% CI bound per pair (NaN to omit the whisker).
    :param ci_high: Upper 95% CI bound per pair (NaN to omit the whisker).
    :param pvalues: P-value per pair used for the row annotation (e.g. Holm-adjusted). Annotated as ``p<0.001`` below
        that, since a bootstrap p-value can be exactly 0 (no resample crossed) and is only bounded from above.
    :param significant: Boolean mask marking pairs significant after correction.
    :param xlabel: X-axis label (e.g. ``'Δ macro AUROC (1st − 2nd)'``); names the metric, so no title is needed.
    :param title: Panel title; omitted when ``None``.
    :param slot: Panel width slot (see :func:`panel_size`); height grows with the number of model pairs.
    :param out_path: If given, save there; otherwise ``plt.show()``.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=panel_size(slot, height_mm=18 + 5.0 * n), constrained_layout=True)
    ypos = np.arange(n)[::-1]  # first label at the top

    for i, y in enumerate(ypos):
        color = 'firebrick' if significant[i] else 'gray'
        lo, hi = ci_low[i], ci_high[i]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=color, lw=1.0, zorder=1)
        ax.plot(deltas[i], y, 'o', color=color, ms=2.5, zorder=2)

    ax.axvline(0, ls='--', color='black', alpha=0.5)

    # Annotate each row with its p-value + stars, right-aligned in a reserved margin past the data.
    finite = [v for v in np.concatenate([deltas, ci_low, ci_high]) if np.isfinite(v)]
    xmin, xmax = min(finite), max(finite)
    pad = 0.08 * (xmax - xmin or 1.0)
    x_text = xmax + pad
    for i, y in enumerate(ypos):
        p = pvalues[i]
        if not np.isfinite(p):
            label = 'p=n/a'
        else:
            # Bounded below: a bootstrap p can underflow to 0, which '%.3g' would print as the false claim 'p=0'.
            label = f'p<0.001{p_stars(p)}' if p < 0.001 else f'p={p:.3g}{p_stars(p)}'
        ax.text(x_text, y, label, ha='left', va='center', fontsize=SMALL_SIZE,
                color='firebrick' if significant[i] else 'dimgray')
    ax.set_xlim(xmin - pad, x_text + 3.5 * pad)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=SMALL_SIZE)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    handles = [Line2D([0], [0], marker='o', color='firebrick', ls='', ms=2.5, label='significant (Holm)'),
               Line2D([0], [0], marker='o', color='gray', ls='', ms=2.5, label='n.s.')]
    ax.legend(handles=handles, loc='upper left', fontsize=SMALL_SIZE)

    if out_path is not None:
        fig.savefig(out_path)
    else:
        plt.show()
    plt.close(fig)
