"""Pairwise significance testing between trained models on the external test split.

Answers "do these two models actually differ?" — the question overlapping per-model CIs (from
:func:`ml.evaluate.bootstrap_ci_metrics`) cannot settle. Compares the models trained on different feature sources
(cpg vs markers vs stacked) for a task: every model is scored on the same external test set, so predictions
are paired by ``slideId`` and the comparison is a *paired* one:

* **classification** compares macro-OvR AUROC with a paired difference bootstrap
  (:func:`ml.stats.paired_bootstrap_auroc_diff`) and each per-class one-vs-rest AUROC with DeLong's test
  (:func:`ml.stats.delong_auroc_diff`) — both give a CI on ΔAUROC plus a p-value;
* **ordinal** compares macro-averaged mean absolute error with a paired difference bootstrap
  (:func:`ml.stats.paired_bootstrap_mae_diff`), likewise clustered on patient — the macro average is taken over
  true-label groups, which no analytic paired test addresses.

All pairwise comparisons for the task are run and their p-values corrected with Holm-Bonferroni
(:func:`ml.stats.holm_correction`) within each family (see :func:`_family`): the metric family — the macro/aggregate
primary endpoint (``auroc_macro`` / ``mae_macro``) on its own, the per-class one-vs-rest AUROCs together — crossed
with the contrast family — comparisons against the stacked ensemble ("does stacking help?", corrected together)
separated from base-vs-base (e.g. markers-vs-cpg). Reads the ``test_predictions__*.csv`` files written by
:mod:`ml.evaluate`; writes one tidy table plus a Δ forest plot per metric.
"""
import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import COMPARISON_DIR, COMPARISON_PLOTS, PREDICTIONS_DIR
from ..visualization import plot_metric_diff_forest
from . import stats
from .cv import _derive_groups
from .evaluate import _y_pred_y_prob_from_results
from .features import SOURCE_LABELS
from .tasks import TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

_PREFIX = 'test_predictions__model__'


def _model_label(path: Path) -> str:
    """Pipeline id for a predictions file: its stem with the ``test_predictions__model__`` prefix stripped."""
    stem = path.stem
    return stem[len(_PREFIX):] if stem.startswith(_PREFIX) else stem


def _short_label(label: str) -> str:
    """Display name for a model on a plot: its feature source, which is what differs between the compared models
    (same task, one model per source). An unknown source falls back to the pipeline id minus the shared ``<task>__``
    prefix, so the filter/reducer/model tokens stay visible."""
    parts = label.split('__')
    return SOURCE_LABELS.get(_feature_source(label)) or ('__'.join(parts[1:]) if len(parts) > 1 else label)


def _discover(task: str) -> list[Path]:
    """External-test prediction files for a task (one per feature source), sorted for deterministic pairing."""
    return sorted(PREDICTIONS_DIR.glob(f'{_PREFIX}{task}__*.csv'))


def _load(path: Path) -> pd.DataFrame:
    """Read a predictions CSV indexed by (string) ``slideId``."""
    df = pd.read_csv(path, index_col='slideId')
    df.index = df.index.astype(str)
    return df


def _align(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both prediction frames to their shared ``slideId`` set, in a common order."""
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError('The two prediction files share no slideIds.')
    a, b = df_a.loc[common], df_b.loc[common]
    if (a['y_true'].to_numpy() != b['y_true'].to_numpy()).any():
        logger.warning('y_true disagrees between the two files — are they the same task/split?')
    return a, b


def _compare_pair(task: Task, df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run the task-appropriate paired test on one aligned model pair, keyed by output metric name."""
    a, b = _align(df_a, df_b)
    y_true = a['y_true'].to_numpy()
    # Patient IDs (aligned to the shared slideId index) cluster the macro-AUROC / macro-MAE bootstraps so same-patient
    # slides are resampled together; the analytic DeLong path below has no clustered form and stays slide-level.
    groups = _derive_groups(a.index).to_numpy()

    if task.name == 'classification':
        _, prob_a = _y_pred_y_prob_from_results(a, task)
        _, prob_b = _y_pred_y_prob_from_results(b, task)
        y_true = y_true.astype(int)
        class_names = tuple(task.class_names or ())
        return {
            **stats.paired_bootstrap_auroc_diff(y_true, prob_a, prob_b, groups=groups),   # auroc_macro (clustered)
            **stats.delong_auroc_diff(y_true, prob_a, prob_b, class_names),               # auroc_<class> (slide-level)
        }

    # ordinal: test the macro MAE that is reported (matches test_metrics.json) — a group average, so bootstrapped.
    y_pred_a, _ = _y_pred_y_prob_from_results(a, task)
    y_pred_b, _ = _y_pred_y_prob_from_results(b, task)
    return stats.paired_bootstrap_mae_diff(                                        # mae_macro (clustered)
        y_true, y_pred_a, y_pred_b, task.metric_fns['mae_macro'], groups=groups,
    )


def _better(metric: str, delta: float, model_a: str, model_b: str, significant: bool) -> str:
    """Name the favoured model for a significant row, else ``'ns'``. AUROC is higher-better, MAE lower-better."""
    if not significant or not np.isfinite(delta) or delta == 0:
        return 'ns'
    a_wins = (delta < 0) if metric.startswith('mae') else (delta > 0)
    return model_a if a_wins else model_b


# The feature source built on top of the base learners (:class:`ml.features.StackedFeatures`). Comparisons that
# involve it answer "does stacking improve on its base learners?" and form a Holm family separate from the
# base-vs-base contrast, matching how the two questions are asked and interpreted separately.
ENSEMBLE_SOURCE = 'stacked'


def _feature_source(label: str) -> str:
    """Feature-source token of a model label (``<task>__<features>__<filter>__<reducer>__<model>`` → ``<features>``)."""
    parts = label.split('__')
    return parts[1] if len(parts) > 1 else label


def _metric_family(metric: str) -> str:
    """Metric half of the Holm family. The macro/aggregate metrics — the primary endpoints (``auroc_macro``,
    ``mae_macro``) — are each their own label; the per-class one-vs-rest AUROCs (``auroc_<class>``) share a single
    secondary label, so a "class X differs" claim is corrected across every class examined rather than per class.
    ``auroc_macro`` is the unweighted mean of the per-class AUROCs, so keeping it separate also avoids correcting a
    quantity against its own components."""
    return metric if metric.endswith('_macro') else f'{metric.split("_")[0]}_perclass'


def _metric_name(metric: str) -> str:
    """Prose name for an output metric, for the plot axis label: ``auroc_macro`` → ``'macro AUROC'`` (as in
    :mod:`ml.importance`), ``auroc_<class>`` → ``'<class> AUROC'`` (the one-vs-rest AUROC for that class)."""
    stat, _, rest = metric.partition('_')
    stat = stat.upper()
    return f'{rest} {stat}' if rest else stat


def _family(metric: str, model_a: str, model_b: str) -> str:
    """Holm family for one comparison row: the metric family (:func:`_metric_family`) crossed with the contrast
    family. Comparisons involving :data:`ENSEMBLE_SOURCE` form the "does stacking beat its base learners?" family
    (so the two stacked-vs-base tests are corrected together); comparisons among the base sources (e.g.
    markers-vs-cpg) form a separate family. Each resulting family spans its pairs within a single metric."""
    contrast = 'vs_ensemble' if ENSEMBLE_SOURCE in (_feature_source(model_a), _feature_source(model_b)) else 'base'
    return f'{_metric_family(metric)}|{contrast}'


def compare(task: Task, paths: list[Path], alpha: float = 0.05) -> pd.DataFrame:
    """Run every pairwise comparison over ``paths`` and return a tidy significance table.

    :param task: Task definition (drives the test and the metric semantics).
    :param paths: External-test ``test_predictions__*.csv`` files to compare (all unordered pairs are tested).
    :param alpha: Significance threshold applied to the Holm-corrected p-values.
    :return: One row per (model pair × output metric) with the test used, point estimates, Δ, CI, statistic, raw/Holm
        p, and the favoured model. Holm correction is applied within each family (:func:`_family`): the metric family
        (macro alone / per-class together) crossed with the contrast family (vs the stacked ensemble / base-vs-base).
    """
    frames = {_model_label(p): _load(p) for p in paths}
    labels = list(frames)
    logger.info('[%s] comparing %d models across %d pairs.', task.name, len(labels),
                len(labels) * (len(labels) - 1) // 2)

    rows = []
    for a, b in combinations(labels, 2):
        for metric, res in _compare_pair(task, frames[a], frames[b]).items():
            rows.append({'metric': metric, 'model_a': a, 'model_b': b, **res})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm-Bonferroni within each family (see _family): the metric family (macro alone / per-class together) crossed
    # with the contrast family (comparisons vs the stacked ensemble, corrected together, vs base-vs-base on its own).
    family = [_family(m, a, b) for m, a, b in zip(df['metric'], df['model_a'], df['model_b'])]
    df['p_value_holm'] = df.groupby(family)['p_value'].transform(stats.holm_correction)
    df['significant'] = df['p_value_holm'] < alpha
    df['better'] = [_better(m, d, a, b, s) for m, d, a, b, s
                    in zip(df['metric'], df['delta'], df['model_a'], df['model_b'], df['significant'])]

    ordered = ['metric', 'test', 'model_a', 'model_b', 'value_a', 'value_b', 'delta', 'ci_low', 'ci_high',
               'statistic', 'p_value', 'p_value_holm', 'significant', 'better', 'n']
    return df.reindex(columns=[c for c in ordered if c in df.columns])


def _plot(df: pd.DataFrame, task: Task, plots_dir: Path, stem: str) -> None:
    """One Δ forest plot per output metric present in the table, annotated with the test used and Holm-adjusted p."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, sub in df.groupby('metric'):
        labels = [f'{_short_label(a)} vs {_short_label(b)}' for a, b in zip(sub['model_a'], sub['model_b'])]
        # Test and sample size are constant within a metric group (same test set, same test), so they go to the log
        # for the figure caption rather than onto every composited panel; the metric is already on the x-axis label.
        test = sub['test'].iloc[0]
        n = int(sub['n'].iloc[0])
        out_path = plots_dir / f'comparison__{stem}__{metric}.pdf'
        plot_metric_diff_forest(
            labels, sub['delta'].to_numpy(), sub['ci_low'].to_numpy(), sub['ci_high'].to_numpy(),
            sub['p_value_holm'].to_numpy(), sub['significant'].to_numpy(),
            xlabel=f'Δ {_metric_name(metric)} (1st − 2nd)', out_path=out_path,
        )
        logger.info('[%s] wrote %s (%s, Holm-adjusted p, n=%d)', task.name, out_path, test, n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), required=True,
                        help='Task to compare models for (compares the models across feature sources).')
    parser.add_argument('--predictions', type=Path, nargs='+', default=None,
                        help='Explicit list of test_predictions__*.csv files to compare, overriding discovery.')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance threshold on the Holm-corrected p-values (default: 0.05).')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--plots-dir', type=Path, default=COMPARISON_PLOTS)
    args = parser.parse_args()

    task = TASKS[args.task]
    paths = args.predictions or _discover(args.task)
    if len(paths) < 2:
        raise SystemExit(f'Need at least 2 prediction files to compare; found {len(paths)}.')

    df = compare(task, paths, alpha=args.alpha)

    out = args.out or COMPARISON_DIR / f'comparison__{args.task}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info('[%s] wrote %s (%d rows)', task.name, out, len(df))

    _plot(df, task, args.plots_dir, args.task)


if __name__ == '__main__':
    main()
