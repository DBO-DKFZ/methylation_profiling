"""Descriptive correlation of the CpG-derived markers against each task's target, for both tasks.

Separate from the ML predictor: this quantifies how the individual markers (Horvath EAA, EpiSCORE cell-type
fractions, CNV burden) track the target on their own, using classical group statistics rather than a model.
It runs for both :mod:`ml.tasks` targets, adapting the tests to each target's measurement level:

* **ordinal** (therapeutic group) — Spearman ρ on the ordinal group code, Kruskal-Wallis across groups, and (only if
  Kruskal-Wallis is significant) Holm-adjusted pairwise Mann-Whitney U. The cross-marker summary is a
  signed Spearman-ρ heatmap.
* **classification** (IM/NIM/NV) — the diagnosis is nominal, so Spearman is undefined and skipped; the group
  comparison is Kruskal-Wallis + conditional pairwise Mann-Whitney U. The cross-marker summary is a pairwise
  Mann-Whitney U effect-size heatmap (group pairs x markers), which unlike Kruskal-Wallis ε² is directional.

The tests delegate to :mod:`ml.stats`; the per-marker omnibus tests (Spearman, Kruskal-Wallis) are Holm-corrected
across the marker set, and the pairwise post-hoc is Holm-corrected within each marker.

The ordinal analysis uses the therapeutic groups the model predicts (``THERAPEUTIC_GROUPS``: ``IIB/IIC`` and
``IIIA/IIIB/IIIC`` collapsed), and both tasks run on the **training split** only (non-test-clinic samples), leaving the
external test set untouched by exploratory analysis. The markers are loaded via the canonical loaders in
:mod:`ml.features.markers` (whole-cohort base values, not the fold-restricted :class:`~ml.features.markers.MarkerFeatures`
swaps), so the correlations describe exactly the feature definitions the ML model consumes.
"""
import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import CORRELATION_DIR, CORRELATION_PLOTS, DIAGNOSIS_COLORS, THERAPEUTIC_COLORS, _cfg_path
from ..visualization import p_stars, plot_grouped_boxplots, plot_marker_heatmap, plot_pairwise_mwu_heatmap
from . import stats
from .ajcc import grouped_ajcc_labels, load_ajcc_metadata
from .features.markers import (
    CNV_COLS, EPISCORE_COLS, HORVATH_COLS, MARKER_NAMES,
    _load_cnv_base, _load_episcore, _load_horvath_eaa, _split_slide_ids,
)
from .tasks import TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# Facet grid for the per-modality boxplots. Three columns puts EpiSCORE's nine cell types on a 3x3 grid; every
# modality shares one facet aspect so a marker figure's panels keep the same shape. The value is the largest that
# still fits a marker figure (four facet rows plus the summary heatmap, see panels.py) on one page.
FACET_NCOLS = 3
FACET_ASPECT = 0.42


# Marker modality -> (its columns, shared y-axis label for the grouped facet), reusing the canonical column blocks so
# a marker added in one place appears here too. Each modality becomes one faceted plot (a panel per marker column).
SOURCES: list[tuple[str, list[str], str]] = [
    ('horvath_eaa', HORVATH_COLS, 'Epigenetic age acceleration'),
    ('episcore', EPISCORE_COLS, 'Estimated fraction'),
    ('cnv', CNV_COLS, 'CNV burden'),
]


# ---------------------------------------------------------------------------
# Task-aware loading
# ---------------------------------------------------------------------------

def _load_group_metadata(task: Task) -> tuple[pd.DataFrame, list[str], bool]:
    """Grouping variable for ``task``, indexed by ``slideId``.

    :param task: Task definition.
    :return: ``(meta, order, is_ordinal)`` where ``meta`` has a ``group`` column (plus ``group_numeric`` for the
        ordinal task), ``order`` is the group labels in display order, and ``is_ordinal`` flags whether a signed
        rank correlation is meaningful (only the ordinal therapeutic group target).
    """
    if task.name == 'ordinal':
        meta = load_ajcc_metadata()
        meta.index = meta.index.astype(str)
        grouped = grouped_ajcc_labels()
        meta = meta.rename(columns={'therapeutic_group': 'group_numeric'})
        meta['group'] = meta['group_numeric'].map(grouped)  # therapeutic groups, matching the ordinal model
        order = [grouped[code] for code in sorted(grouped)]
        return meta[['group', 'group_numeric']], order, True

    # classification: nominal diagnosis over all three classes (order/colors from the task definition).
    meta = pd.read_csv(_cfg_path('meta_data'), index_col='slideId')
    meta.index = meta.index.astype(str)
    order = list(task.class_names)
    meta = meta[meta[task.target_col].isin(order)]
    meta = meta.rename(columns={task.target_col: 'group'})
    return meta[['group']], order, False


def _load_markers(task: Task) -> pd.DataFrame:
    """Whole-cohort base marker values, joined on ``slideId``.

    Uses the shared loaders from :mod:`ml.features.markers` (Horvath ``eaa_base`` — the training-cohort residual,
    identical across tasks — plus EpiSCORE fractions and CNV burden), i.e. the same feature definitions the ML
    pipeline consumes, without the per-fold leakage swaps that only matter inside CV.

    :param task: Task definition — selects the per-task Horvath EAA file.
    :return: DataFrame indexed by ``slideId`` with the :data:`SOURCES` columns.
    """
    horvath = _load_horvath_eaa(task)[['eaa_base']].rename(columns={'eaa_base': HORVATH_COLS[0]})
    return horvath.join(_load_episcore(), how='outer').join(_load_cnv_base(), how='outer')


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(
    df: pd.DataFrame, marker_col: str, is_ordinal: bool, numeric_col: str | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Omnibus group statistics for one marker against the task's grouping variable.

    Runs Spearman (ordinal task only) and Kruskal-Wallis (with an ε² effect size). The pairwise Mann-Whitney U
    post-hoc is *not* run here: :func:`run_all` runs it after Holm-correcting the Kruskal-Wallis p across the marker
    set, so the post-hoc is gated on the same across-marker adjusted p that every output reports. The omnibus ``p_adj``
    is likewise filled by :func:`run_all`.

    :param df: Merged frame with ``marker_col``, ``group``, and (ordinal) ``numeric_col``.
    :param marker_col: Marker column to test.
    :param is_ordinal: Whether to compute Spearman ρ on ``numeric_col``.
    :param numeric_col: Ordinal group code column (required when ``is_ordinal``).
    :return: ``(rows, grouped)`` — one dict per omnibus test (keys ``test, group1, group2, statistic, p, p_adj,
        effect_size, n``) and the per-group value arrays (groups with ``n >= 2``), reused by the gated post-hoc.
    """
    cols = [marker_col, 'group'] + ([numeric_col] if numeric_col else [])
    clean = df[cols].dropna()
    n_total = len(clean)
    results: list[dict] = []

    if is_ordinal:
        rho, p_spearman = spearmanr(clean[numeric_col], clean[marker_col])
        results.append({'test': 'spearman', 'group1': None, 'group2': None,
                        'statistic': rho, 'p': p_spearman, 'p_adj': None, 'effect_size': None, 'n': n_total})

    grouped = {label: g[marker_col].values for label, g in clean.groupby('group') if len(g) >= 2}
    kw = stats.kruskal_effect_size(*grouped.values())
    results.append({'test': 'kruskal_wallis', 'group1': None, 'group2': None,
                    'statistic': kw['statistic'], 'p': kw['p_value'], 'p_adj': None,
                    'effect_size': kw['effect_size'], 'n': n_total})

    return results, grouped


def _fmt_p(p: float) -> str:
    """Compact p-value string for plot annotations, with the shared significance stars (:func:`visualization.p_stars`)
    so a facet reads as significant without decoding the exponent."""
    if pd.isna(p):
        return 'n/a'
    return f'{p:.2e}{p_stars(p)}' if p < 0.001 else f'{p:.4f}{p_stars(p)}'


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

def run_all(task: Task, results_dir: Path, plots_dir: Path) -> None:
    """Run the correlation analysis for one task, writing the CSV, per-marker boxplots, and the summary heatmap.

    :param task: Task definition.
    :param results_dir: Directory for ``statistical_tests.csv`` (created if missing).
    :param plots_dir: Directory for the ``<source>_<marker>.pdf`` boxplots and ``heatmap.pdf`` (created if missing).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta, order, is_ordinal = _load_group_metadata(task)
    merged = meta.join(_load_markers(task), how='inner')
    # Restrict to the training split (non-test-clinic) so the external test set is untouched by exploratory analysis.
    merged = merged[merged.index.isin(_split_slide_ids('train'))]
    logger.info('[%s] %d training-split samples after joining markers with group metadata', task.name, len(merged))

    xlabel = 'Therapeutic group' if is_ordinal else 'Diagnosis'
    # Boxplot colors are keyed by display label, so map each therapeutic group code to its grouped AJCC label.
    colors = ({label: THERAPEUTIC_COLORS[code] for code, label in grouped_ajcc_labels().items()}
              if is_ordinal else DIAGNOSIS_COLORS)
    numeric_col = 'group_numeric' if is_ordinal else None

    # Pass 1: the omnibus tests (Spearman/Kruskal-Wallis) plus the per-group arrays for every marker, so the omnibus
    # p-values can be Holm-corrected across the full marker set before anything is gated, written, or plotted.
    per_marker = []  # (source, marker, omnibus_rows, grouped)
    for source, cols, _ylabel in SOURCES:
        for col in cols:
            rows, grouped = run_tests(merged, col, is_ordinal, numeric_col)
            per_marker.append((source, col, rows, grouped))

    # Holm-correct each omnibus test across the full marker set (one family per test type per task), writing the
    # adjusted p back onto the rows before the pairwise post-hoc is gated on it.
    for test_name in ('spearman', 'kruskal_wallis'):
        rows = [r for _, _, marker_rows, _ in per_marker for r in marker_rows if r['test'] == test_name]
        for r, p_adj in zip(rows, stats.holm_correction([r['p'] for r in rows])):
            r['p_adj'] = float(p_adj)

    # Pass 2: assemble the table in marker order — omnibus rows, then the pairwise Mann-Whitney post-hoc gated on the
    # across-marker Holm-adjusted Kruskal-Wallis p, so the post-hoc agrees with the significance every output reports.
    # The pairwise family stays Holm-adjusted within its own marker by ml.stats.pairwise_mannwhitney.
    all_results = []
    for source, col, marker_rows, grouped in per_marker:
        for r in marker_rows:
            all_results.append({'source': source, 'marker': col, **r})
        kw = next(r for r in marker_rows if r['test'] == 'kruskal_wallis')
        if pd.notna(kw['p_adj']) and kw['p_adj'] < 0.05:
            for r in stats.pairwise_mannwhitney(grouped, order):
                all_results.append({'source': source, 'marker': col, 'test': 'mann_whitney_u',
                                    'group1': r['group1'], 'group2': r['group2'], 'statistic': r['statistic'],
                                    'p': r['p_value'], 'p_adj': r['p_adj'], 'effect_size': r['effect_size'],
                                    'n': r['n']})

    results_df = pd.DataFrame(all_results)
    csv_path = results_dir / 'statistical_tests.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info('[%s] wrote %s', task.name, csv_path)

    # Pass 2: one faceted boxplot per modality (a panel per marker), annotated with the across-marker Holm-adjusted
    # Kruskal-Wallis p so the panels agree with the CSV and heatmap. Ordinal facets add Spearman ρ, which carries a
    # direction the boxes alone do not; classification carries no effect size here, its summary heatmap being one.
    kw = results_df[results_df['test'] == 'kruskal_wallis'].set_index(['source', 'marker'])
    sp = results_df[results_df['test'] == 'spearman'].set_index(['source', 'marker']) if is_ordinal else None
    for source, cols, ylabel in SOURCES:
        source_annotations: dict[str, dict[str, str]] = {}
        for col in cols:
            # Terse keys: the box sits inside a facet as narrow as 36 mm. Which test the p comes from
            # (Kruskal-Wallis, Holm-adjusted across markers) belongs in the figure caption.
            annotations = {'p': _fmt_p(kw.loc[(source, col), 'p_adj'])}
            if is_ordinal:
                annotations = {'ρ': f"{sp.loc[(source, col), 'statistic']:.3f}", **annotations}
            source_annotations[col] = annotations

        # Width slot follows the facet count so every facet lands at 'third' width: a single-marker source (CNV) is a
        # third-width panel, EpiSCORE's nine cell types a full-width one.
        slot = {1: 'third', 2: 'two_thirds'}.get(len(cols), 'full')
        plot_grouped_boxplots(
            merged, marker_cols=cols, group_col='group', order=order, xlabel=xlabel, ylabel=ylabel,
            annotations=source_annotations, colors=colors, slot=slot, ncols=FACET_NCOLS,
            facet_aspect=FACET_ASPECT, names=MARKER_NAMES, out_path=plots_dir / f'{source}.pdf',
        )

    # Cross-marker summary heatmap. Markers run along x in both variants, so the panel is a wide, short strip that
    # frees the vertical space the EpiSCORE facets need above it, and marker name alone labels the x axis
    # ("<source> / <marker>" is too long for a rotated tick label, and no marker name repeats across sources).
    heatmap_path = plots_dir / 'heatmap.pdf'
    if is_ordinal:
        # One signed Spearman ρ per marker; stars use the across-marker Holm-adjusted p, matching the cross-marker
        # scope of the strip.
        rows = results_df[results_df['test'] == 'spearman'].set_index('marker').rename(index=MARKER_NAMES)
        plot_marker_heatmap(
            rows['statistic'], rows['p_adj'], value_label='Spearman ρ', diverging=True, slot='two_thirds',
            horizontal=True, out_path=heatmap_path,
        )
    else:
        # The nominal diagnosis has no signed omnibus statistic, so the summary is the pairwise Mann-Whitney U effect
        # size per group pair — directional, unlike Kruskal-Wallis ε². Stars are the within-marker Holm-adjusted
        # pairwise p, the family the post-hoc itself was corrected over. Markers whose omnibus test did not clear the
        # gate above have no post-hoc rows and stay blank, so the panel and the CSV report exactly the same tests.
        rows = results_df[results_df['test'] == 'mann_whitney_u']
        rows = rows.assign(pair=rows['group1'] + ' vs ' + rows['group2'])
        pair_order = [f'{g1} vs {g2}' for g1, g2 in combinations(order, 2)]
        values, pvalues = (rows.pivot(index='pair', columns='marker', values=v)
                           .reindex(index=pair_order, columns=[c for _, cols, _ in SOURCES for c in cols])
                           .rename(columns=MARKER_NAMES)
                           for v in ('effect_size', 'p_adj'))
        plot_pairwise_mwu_heatmap(values, pvalues, slot='two_thirds', out_path=heatmap_path)
    logger.info('[%s] wrote %s', task.name, heatmap_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default=None,
                        help='Task to analyse (default: run both).')
    parser.add_argument('--results-dir', type=Path, default=CORRELATION_DIR)
    parser.add_argument('--plots-dir', type=Path, default=CORRELATION_PLOTS)
    args = parser.parse_args()

    for name in ([args.task] if args.task else list(TASKS)):
        run_all(TASKS[name], args.results_dir / name, args.plots_dir / name)


if __name__ == '__main__':
    main()
