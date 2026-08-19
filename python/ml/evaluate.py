"""Evaluate a trained artefact on the external test split.

Task-aware: branches on ``artifact['task']`` (stamped by :mod:`ml.train`) so the same script covers both classification
and ordinal pipelines. Computes the task's headline metrics (with bootstrap CIs) and emits task-appropriate plots.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from ..config import CLASSIFIER_PLOTS, PREDICTIONS_DIR, RANDOM_STATE
from ..visualization import plot_confusion_matrix, plot_per_group_mae, plot_roc_curves
from .ajcc import grouped_ajcc_labels
from .cv import _derive_groups, build_predictions_df
from .stats import _group_rows
from .features import SOURCE_LABELS, build as build_feature_source
from .tasks import THERAPEUTIC_GROUPS, TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def bootstrap_ci_metrics(
    task: Task,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    groups: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute every metric in ``task.metric_fns`` (and each ``task.per_class_metric_fns``) with a 95% bootstrap CI.

    Resamples are shared across metrics so their CIs are drawn from the same bootstrap distribution. Each entry in
    ``task.per_class_metric_fns`` returns one value per class and is expanded into ``<base>_<class>`` metrics that draw
    from the same resamples. Resamples that crash a particular metric (e.g. a class missing in the bootstrap sample
    makes ``roc_auc_score`` fail) are skipped for that metric only — they still count toward other metrics.

    When ``groups`` is given the bootstrap is patient-clustered (:func:`ml.stats._group_rows`): whole patients are
    drawn with replacement so their correlated slides stay together rather than being resampled as independent
    observations. Without ``groups`` it is the ordinary per-row bootstrap.

    :param task: Task definition; iterates ``task.metric_fns`` and ``task.per_class_metric_fns``.
    :param y_true: 1-D ground-truth array.
    :param y_pred: 1-D predicted-label / regressor-output array, aligned to ``y_true``.
    :param y_prob: ``(n_samples, n_classes)`` probability matrix, or ``None`` for ordinal regression.
    :param groups: Optional 1-D cluster labels (patient IDs) aligned to ``y_true``; clusters the resampling.
    :param n_bootstrap: Number of valid bootstrap iterations (default: 1000).
    :param random_state: Seed for the bootstrap RNG.
    :return: ``{metric_name: {'value': float, 'ci_low': float, 'ci_high': float}}``.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = tuple(task.class_names or ())
    group_rows = _group_rows(np.arange(n) if groups is None else np.asarray(groups))
    n_groups = len(group_rows)

    results: dict[str, dict[str, float]] = {}
    for name, fn in task.metric_fns.items():
        results[name] = {'value': float(fn(pd.Series(y_true), y_pred, y_prob))}
    for base, fn in task.per_class_metric_fns.items():
        for cls, v in zip(class_names, fn(pd.Series(y_true), y_pred, y_prob)):
            results[f'{base}_{cls}'] = {'value': float(v)}

    samples: dict[str, list[float]] = {name: [] for name in results}
    for _ in range(2 * n_bootstrap):
        if all(len(s) >= n_bootstrap for s in samples.values()):
            break
        idx = np.concatenate([group_rows[g] for g in rng.integers(0, n_groups, size=n_groups)])
        y_t = pd.Series(y_true[idx]).reset_index(drop=True)
        y_p = y_pred[idx]
        y_pr = y_prob[idx] if y_prob is not None else None
        for name, fn in task.metric_fns.items():
            if len(samples[name]) >= n_bootstrap:
                continue
            try:
                samples[name].append(float(fn(y_t, y_p, y_pr)))
            except (ValueError, IndexError):
                continue
        for base, fn in task.per_class_metric_fns.items():
            names = [f'{base}_{cls}' for cls in class_names]
            if all(len(samples[nm]) >= n_bootstrap for nm in names):
                continue
            try:
                arr = fn(y_t, y_p, y_pr)
            except (ValueError, IndexError):
                continue
            for nm, v in zip(names, arr):
                if len(samples[nm]) >= n_bootstrap:
                    continue
                samples[nm].append(float(v))

    for name, arr in samples.items():
        finite = np.array([v for v in arr if not np.isnan(v)], dtype=float)
        if len(finite) >= 2:
            results[name]['ci_low'] = float(np.percentile(finite, 2.5))
            results[name]['ci_high'] = float(np.percentile(finite, 97.5))
        else:
            results[name]['ci_low'] = float('nan')
            results[name]['ci_high'] = float('nan')
    return results


def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    artifact: dict,
    task: Task,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Apply the artefact's pipeline (filter → reducer → predict) and emit per-sample predictions.

    :param X: Feature matrix in the original (pre-filter) feature space.
    :param y: Target aligned to ``X``.
    :param artifact: Dict produced by :mod:`ml.train` with keys ``model``, ``selected_cpgs``, ``reducer``.
    :param task: Task definition; drives ``predict`` vs ``predict_proba`` and the output schema.
    :param out_path: If given, write the predictions DataFrame as CSV.
    :return: DataFrame indexed by sample ID with ``y_true`` and either ``prob_<class>`` (classification) or
        ``prediction`` (ordinal) columns; ordinal models exposing ``predict_proba`` also get ``prob_<class>`` columns.
    """
    X = X[artifact['selected_cpgs']]
    logger.info('After CpG filter: %d features.', X.shape[1])
    # Capture the slide-ID index now — a reducer may return a bare NumPy array, dropping it.
    sample_index = X.index

    if artifact['reducer'] is not None:
        X = artifact['reducer'].transform(X)
        logger.info('After reducer: %d features.', X.shape[1])

    model = artifact['model']
    pred_fn = getattr(model, task.predict_method)

    if task.oof_schema == 'probs':
        y_prob = pred_fn(X)
        classes = np.asarray(model.classes_)
        results = build_predictions_df(
            sample_index, y, task, classes=classes, y_prob=y_prob, include_fold=False,
        )
    else:
        y_pred = pred_fn(X)
        # Ordinal: keep probabilities (when available) as extra columns for stacking; metrics use ``prediction``.
        y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        classes = np.asarray(model.classes_) if y_prob is not None else None
        results = build_predictions_df(
            sample_index, y, task, y_pred=y_pred, classes=classes, y_prob=y_prob, include_fold=False,
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index_label='slideId')
        logger.info('Predictions saved to %s.', out_path)

    return results


def _y_pred_y_prob_from_results(results: pd.DataFrame, task: Task) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Reconstruct ``(y_pred, y_prob)`` from a predictions DataFrame in either OOF schema."""
    if task.oof_schema == 'prediction':
        return results['prediction'].to_numpy(), None
    prob_cols = sorted((c for c in results.columns if c.startswith('prob_')),
                       key=lambda c: int(c.split('_')[1]))
    y_prob = results[prob_cols].to_numpy()
    classes = np.array([int(c.split('_')[1]) for c in prob_cols])
    y_pred = classes[np.argmax(y_prob, axis=1)]
    return y_pred, y_prob


def _analyze_classification(
    results: pd.DataFrame,
    task: Task,
    metrics: dict[str, dict[str, float]],
    out_dir: Path,
    label: Optional[str] = None,
) -> None:
    """Classification-specific plots: ROC curves + confusion matrix using ``task.class_names``.

    Per-class AUROC point estimates are read from ``metrics`` (populated by :func:`bootstrap_ci_metrics`) so the
    curve labels match the reported values. ``label``, when given, titles both panels (see :func:`analyze`).
    """
    class_names = list(task.class_names or [])
    y_true = results['y_true'].to_numpy().astype(int)
    y_pred, y_prob = _y_pred_y_prob_from_results(results, task)

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    mean_fpr = np.linspace(0, 1, 200)
    mean_tprs = []
    for i, name in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        fpr_dict[name] = fpr_i
        tpr_dict[name] = tpr_i
        auc_dict[name] = metrics[f'auroc_{name}']['value']
        mean_tprs.append(np.interp(mean_fpr, fpr_i, tpr_i))

    fpr_dict['macro'] = mean_fpr
    tpr_dict['macro'] = np.mean(mean_tprs, axis=0)
    auc_dict['macro'] = metrics['auroc']['value']
    plot_roc_curves(fpr_dict, tpr_dict, auc_dict, title=label, out_path=out_dir / 'roc_curves.pdf')

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, title=label, out_path=out_dir / 'confusion_matrix.pdf')


def _analyze_ordinal(
    results: pd.DataFrame,
    task: Task,
    metrics: dict[str, dict[str, float]],
    out_dir: Path,
    label: Optional[str] = None,
) -> None:
    """Ordinal-specific plots: per-group MAE bar chart + rounded group-vs-group confusion matrix, both titled with
    ``label`` when given (see :func:`analyze`)."""
    y_true = results['y_true'].to_numpy()
    y_pred = results['prediction'].to_numpy()
    y_pred_int = np.clip(np.round(y_pred), THERAPEUTIC_GROUPS[0], THERAPEUTIC_GROUPS[-1]).astype(int)
    y_true_int = y_true.astype(int)

    # Reuse the already-computed per-group MAEs and their patient-clustered bootstrap CIs (mae_group_*), so the bars
    # and error bars match test_metrics.json rather than recomputing a plain mean here.
    present = [g for g in THERAPEUTIC_GROUPS if pd.notna(metrics.get(f'mae_group_{g}', {}).get('value', np.nan))]
    per_group = pd.Series({g: metrics[f'mae_group_{g}']['value'] for g in present})
    ci = pd.DataFrame(
        {g: {k: metrics[f'mae_group_{g}'][k] for k in ('ci_low', 'ci_high')} for g in present}
    ).T
    plot_per_group_mae(per_group, ci=ci, title=label, out_path=out_dir / 'per_group_mae.pdf')

    cm = confusion_matrix(y_true_int, y_pred_int, labels=THERAPEUTIC_GROUPS)
    group_labels = grouped_ajcc_labels()
    plot_confusion_matrix(cm, [group_labels[g] for g in THERAPEUTIC_GROUPS], title=label,
                          out_path=out_dir / 'confusion_matrix.pdf')


def analyze(
    results: pd.DataFrame,
    task: Task,
    out_dir: Path,
    label: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """
    Computes ``task.metric_fns`` with bootstrap CIs, writes ``test_metrics.json``, and dispatches to
    the task-specific plotting routine.

    :param results: Per-sample predictions as produced by :func:`evaluate`.
    :param task: Task definition.
    :param out_dir: Directory where plots and ``test_metrics.json`` are saved.
    :param label: Optional model name used as the plot titles; the CLI passes the feature-source display name
        (:data:`ml.features.SOURCE_LABELS`), which is what tells apart the panels composited into a single figure.
    :return: The metrics dict ``{name: {value, ci_low, ci_high}}`` (also persisted to JSON).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = results['y_true'].to_numpy()
    y_pred, y_prob = _y_pred_y_prob_from_results(results, task)

    groups = _derive_groups(results.index).to_numpy()
    metrics = bootstrap_ci_metrics(task, y_true, y_pred, y_prob, groups=groups)

    for name, m in metrics.items():
        logger.info('%s: %.4f [%.4f - %.4f]', name, m['value'], m['ci_low'], m['ci_high'])

    (out_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    logger.info('Metrics saved to %s', out_dir / 'test_metrics.json')

    if task.name == 'classification':
        _analyze_classification(results, task, metrics, out_dir, label=label)
    elif task.name == 'ordinal':
        _analyze_ordinal(results, task, metrics, out_dir, label=label)
    else:
        raise ValueError(f'No analyze() branch for task {task.name!r}.')

    logger.info('Plots saved to %s', out_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True,
                        help='Path to the artefact produced by ml.train.')
    parser.add_argument('--predictions', type=Path, default=None,
                        help='Where to write the per-sample predictions CSV. Default: <PREDICTIONS_DIR>/'
                             'test_predictions__<artifact stem>.csv.')
    parser.add_argument('--plots-dir', type=Path, default=None,
                        help='Directory for plots and test_metrics.json. Default: <CLASSIFIER_PLOTS>/<artifact stem>.')
    parser.add_argument('--label', default=None,
                        help='Model name to title the plots with. Defaults to the artefact\'s feature-source display '
                             'name (e.g. "CpG-based"), which is what distinguishes the models composited side by side '
                             'in a manuscript panel; pass an empty string for untitled plots.')
    parser.add_argument('--cpg-oof', type=Path, default=None,
                        help='Required for --features stacked: the CpG base model\'s test-set predictions '
                             '(a test_predictions__*.csv). OOF predictions cover training rows only, so external-test '
                             'stacked evaluation uses the base model\'s own test predictions instead.')
    parser.add_argument('--marker-oof', type=Path, default=None,
                        help='Required for --features stacked: the marker base model\'s test-set predictions '
                             '(a test_predictions__*.csv), used as the second view. Must match the --marker-oof '
                             'passed at train time.')
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)

    missing = {'task', 'features'} - set(artifact)
    if missing:
        raise KeyError(
            f'Artifact at {args.artifact} is missing required keys {sorted(missing)}. '
            f'Re-train with the current ml.train (which stamps task/features/...) to evaluate it.'
        )
    task = TASKS[artifact['task']]
    feature_source = build_feature_source(
        artifact['features'],
        args.cpg_oof,
        split='test',
        marker_oof_path=args.marker_oof,
    )
    X_test, y_test = feature_source.load(task)

    stem = args.artifact.stem
    predictions_path = args.predictions or (PREDICTIONS_DIR / f'test_predictions__{stem}.csv')
    out_plots_dir = args.plots_dir or (CLASSIFIER_PLOTS / stem)

    label = args.label if args.label is not None else SOURCE_LABELS.get(artifact['features'])

    results = evaluate(X_test, y_test, artifact, task, out_path=predictions_path)
    analyze(results, task, out_dir=out_plots_dir, label=label)


if __name__ == '__main__':
    main()
