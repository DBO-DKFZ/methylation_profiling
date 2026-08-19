"""Feature importance for the low-dimensional feature sources (markers, stacked).

Both marker features (12 CpG-derived markers) and stacked meta-features (base-learner prediction columns) are already
named and low-dim, with no filter/reducer step, so importance is measured directly on the model's input columns — no
clustering, ChromHMM grouping, or selection enrichment (those only make sense for the ~644k raw-CpG model, which this
module deliberately does not handle).

Permutation importance is computed on the external test split, task-aware (classification and ordinal): the drop in
the task's score when a single feature is shuffled.

It is also reported **grouped** by semantic feature block (see :func:`_feature_groups`): the stacked model's two
base-learner blocks (CpG-view vs marker-view) and the marker model's data modalities (Horvath EAA / EpiScore cell
composition / CNV burden). Grouping matters most for the stacked model — each base learner's probability columns sum
to 1, so shuffling one at a time understates the block (the model recovers the signal from its siblings); permuting
the whole block jointly measures the block's real contribution. Grouped permutation shuffles a block's columns with a
single shared row order (preserving each row's within-block joint distribution).

Like :mod:`ml.evaluate`, the entry point loads a trained artefact, rebuilds its feature source from the stamped
``features``/``task`` keys, and scores the model on the test set.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ..config import IMPORTANCE_DIR, IMPORTANCE_PLOTS, RANDOM_STATE
from ..visualization import plot_feature_importance
from .features import SOURCE_LABELS, build as build_feature_source
from .features.markers import MARKER_NAMES
from .tasks import TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)


def _permutation_scorer(task: Task) -> tuple:
    """Build the permutation-importance scorer for ``task``, reusing the task's own headline metric.

    Returns a plain ``callable(estimator, X, y) -> float`` that calls the model's response method directly and scores
    with the *same* metric function the rest of the pipeline reports (:mod:`ml.tasks`): macro one-vs-rest AUROC for
    classification (``predict_proba``) and macro-averaged MAE for the ordinal therapeutic group task (``predict``). Calling the
    response method directly sidesteps sklearn's string-scorer inference, which gates ``predict_proba`` on
    :func:`sklearn.base.is_classifier` — the project's estimators don't register as classifiers under that check.

    The returned ``greater_is_better`` flag lets callers orient the importance so that a *larger* value always means a
    *more* important feature (score drop for AUROC, score rise for MAE); ``xlabel`` phrases that for the plot axis.

    :param task: Task definition (branches on ``task.oof_schema``).
    :return: ``(scorer, label, xlabel, greater_is_better)``.
    """
    if task.oof_schema == 'probs':
        metric = task.metric_fns['auroc']

        def scorer(estimator, X, y):
            return float(metric(pd.Series(np.asarray(y)), None, estimator.predict_proba(X)))
        return scorer, 'macro AUROC', 'Drop in macro AUROC', True

    metric = task.metric_fns['mae_macro']

    def scorer(estimator, X, y):
        return float(metric(pd.Series(np.asarray(y)), np.asarray(estimator.predict(X)), None))
    return scorer, 'macro MAE', 'Increase in macro MAE', False


def _feature_groups(artifact: dict, feature_names) -> np.ndarray:
    """Map each feature to its semantic group label, keyed on the artefact's feature source.

    * **markers** — group by data modality using the canonical :mod:`ml.features.markers` column blocks: Horvath EAA,
      EpiScore cell composition, CNV burden.
    * **stacked** — group by base learner: two-base-learner artefacts prefix their columns ``cpg_``/``mrk_``, so the
      CpG view is the ``cpg_``-prefixed prediction columns and the marker view is the ``mrk_``-prefixed ones.

    :param artifact: Artefact dict (uses ``artifact['features']``).
    :param feature_names: The model's input feature names (``artifact['selected_cpgs']``).
    :return: Array of group labels aligned to ``feature_names``.
    """
    features = list(feature_names)
    kind = artifact['features']

    if kind == 'markers':
        from .features.markers import CNV_COLS, EPISCORE_COLS, HORVATH_COLS
        modality = {c: 'Horvath EAA' for c in HORVATH_COLS}
        modality.update({c: 'EpiScore (cell composition)' for c in EPISCORE_COLS})
        modality.update({c: 'CNV burden' for c in CNV_COLS})
        return np.array([modality.get(f, 'other') for f in features])

    if kind == 'stacked':
        return np.array(['CpG view' if f.startswith('cpg_') else 'marker view' for f in features])

    raise ValueError(f'No feature grouping defined for features={kind!r}.')


def compute_permutation_importance(
    artifact: dict,
    X: pd.DataFrame,
    y: pd.Series,
    scorer,
    scoring_label: str,
    greater_is_better: bool = True,
    n_repeats: int = 100,
    n_jobs: int = -1,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Per-feature permutation importance for a trained low-dim artefact.

    Permutes each of the model's input columns (``artifact['selected_cpgs']`` — for markers/stacked this is simply the
    full, unfiltered feature set) in turn and measures the resulting change in ``scorer``, averaged over ``n_repeats``
    permutations. ``importance_mean`` is oriented so larger = more important regardless of metric direction: a drop for
    a higher-is-better metric (AUROC), a rise for a lower-is-better one (MAE).

    :param artifact: Dict as produced by :func:`ml.train.train` with keys ``model`` and ``selected_cpgs``.
    :param X: Feature matrix of shape (n_samples, n_features) covering all training-time features.
    :param y: Target aligned to ``X`` (integer class labels for classification, therapeutic group codes for ordinal).
    :param scorer: Callable ``(estimator, X, y) -> float`` returning the raw metric, e.g. from
        :func:`_permutation_scorer`.
    :param scoring_label: Short metric name for logging (e.g. ``'macro AUROC'``).
    :param greater_is_better: Whether higher ``scorer`` values are better; drives the sign so importance stays positive
        for informative features.
    :param n_repeats: Number of permutation repeats per feature (default: 100).
    :param n_jobs: Parallel jobs forwarded to :func:`sklearn.inspection.permutation_importance` (default: -1).
    :param out_path: If given, save the importance DataFrame to this CSV path.
    :return: DataFrame with columns ``feature``, ``importance_mean``, ``importance_std``, sorted descending by
        ``importance_mean``.
    """
    features = list(artifact['selected_cpgs'])
    X_sel = X[features]
    logger.info('Computing permutation importance: %d features, %d samples, n_repeats=%d, scoring=%s',
                len(features), len(X_sel), n_repeats, scoring_label)

    # sklearn's permutation_importance reports baseline - permuted, assuming higher = better; feed it a higher-is-better
    # scorer (negate a lower-is-better metric) so the resulting importance is positive for informative features.
    oriented = scorer if greater_is_better else (lambda est, Xp, yp: -scorer(est, Xp, yp))
    result = permutation_importance(
        artifact['model'], X_sel, y,
        n_repeats=n_repeats, scoring=oriented,
        random_state=RANDOM_STATE, n_jobs=n_jobs,
    )

    df = pd.DataFrame({
        'feature': features,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

    logger.info('Feature permutation importance:\n%s', df.to_string(index=False))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info('Importance table saved to %s.', out_path)

    return df


def compute_grouped_permutation_importance(
    artifact: dict,
    X: pd.DataFrame,
    y: pd.Series,
    group_labels: np.ndarray,
    scorer,
    scoring_label: str,
    greater_is_better: bool = True,
    n_repeats: int = 100,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Group-level permutation importance: shuffle each group's columns jointly and measure the score change.

    All columns of a group are permuted with a *single shared* row order per repeat, so each row's within-group joint
    distribution is preserved (e.g. a base learner's ``prob_*`` simplex stays a valid row) while the group as a whole
    is decorrelated from ``y``. ``importance_mean`` is oriented so larger = the group contributes more on top of
    everything else, regardless of metric direction (a score drop for AUROC, a score rise for MAE).

    :param artifact: Dict as produced by :func:`ml.train.train` with keys ``model`` and ``selected_cpgs``.
    :param X: Feature matrix covering all training-time features.
    :param y: Target aligned to ``X``.
    :param group_labels: Group label per feature, aligned to ``artifact['selected_cpgs']`` (see :func:`_feature_groups`).
    :param scorer: Callable ``(estimator, X, y) -> float`` returning the raw metric, e.g. from
        :func:`_permutation_scorer`.
    :param scoring_label: Short metric name for logging and the ``baseline`` column.
    :param greater_is_better: Whether higher ``scorer`` values are better; drives the sign so importance stays positive
        for informative groups.
    :param n_repeats: Number of permutation repeats per group (default: 100).
    :param out_path: If given, save the grouped importance DataFrame to this CSV path.
    :return: DataFrame with columns ``group``, ``n_features``, ``features``, ``importance_mean``, ``importance_std``,
        ``baseline``, sorted descending by ``importance_mean``. ``baseline`` is the unpermuted metric (``scoring_label``).
    """
    features = np.asarray(artifact['selected_cpgs'])
    model = artifact['model']
    X_ev = X[list(features)]

    baseline = scorer(model, X_ev, y)
    logger.info('Grouped permutation importance: %d groups over %d features, baseline %s=%.4f',
                len(pd.unique(group_labels)), len(features), scoring_label, baseline)

    orig = X_ev.values
    n_samples = len(orig)
    X_perm = X_ev.copy()  # keep column names so the model doesn't warn / mis-order on predict
    rng = np.random.default_rng(RANDOM_STATE)
    sign = 1.0 if greater_is_better else -1.0  # orient so a more important group scores higher

    records = []
    for group in pd.unique(group_labels):
        col_idx = np.where(group_labels == group)[0]
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            perm = rng.permutation(n_samples)
            X_perm.iloc[:, col_idx] = orig[perm][:, col_idx]
            drops[r] = sign * (baseline - scorer(model, X_perm, y))
        X_perm.iloc[:, col_idx] = orig[:, col_idx]  # restore before the next group
        records.append({
            'group': group,
            'n_features': len(col_idx),
            'features': ';'.join(features[col_idx]),
            'importance_mean': float(drops.mean()),
            'importance_std': float(drops.std()),
            'baseline': float(baseline),
        })

    df = pd.DataFrame(records).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    logger.info('Grouped permutation importance:\n%s',
                df[['group', 'n_features', 'importance_mean', 'importance_std']].to_string(index=False))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info('Grouped importance table saved to %s.', out_path)

    return df


def run(
    artifact: dict,
    task: Task,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    stem: str,
    n_repeats: int = 100,
    results_dir: Path = IMPORTANCE_DIR,
    plots_dir: Path = IMPORTANCE_PLOTS,
    label: Optional[str] = None,
) -> None:
    """Compute and plot per-feature and grouped permutation importance for a low-dim artefact, keyed by ``stem``.

    Outputs go to ``<results_dir>/{permutation,permutation_grouped}__<stem>.csv`` and the matching
    ``<plots_dir>/*.pdf``. Groups are the semantic blocks from :func:`_feature_groups`.

    :param artifact: Trained artefact dict (``model``/``selected_cpgs``).
    :param task: Task definition — drives the permutation scorer.
    :param X_test: External-test feature matrix.
    :param y_test: External-test target aligned to ``X_test``.
    :param stem: Artefact stem used to name the output files.
    :param n_repeats: Permutation repeats per feature/group (default: 100).
    :param results_dir: Directory for the CSV tables.
    :param plots_dir: Directory for the PDF plots.
    :param label: Optional model name used as the plot titles; the CLI passes the feature-source display name
        (:data:`ml.features.SOURCE_LABELS`), since in a manuscript panel the file name is no longer visible to tell
        the models apart.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    group_labels = _feature_groups(artifact, artifact['selected_cpgs'])

    # ---- Permutation importance (per-feature + grouped) ----
    scorer, scoring_label, xlabel, greater_is_better = _permutation_scorer(task)
    perm_df = compute_permutation_importance(
        artifact, X_test, y_test, scorer=scorer, scoring_label=scoring_label,
        greater_is_better=greater_is_better, n_repeats=n_repeats,
        out_path=results_dir / f'permutation__{stem}.csv',
    )
    plot_feature_importance(
        # Display names only on the bars (as in the marker correlation figures); the CSV keeps the raw column names.
        perm_df.assign(feature=perm_df['feature'].replace(MARKER_NAMES)),
        top_n=len(perm_df), xlabel=xlabel, title=label,
        out_path=plots_dir / f'permutation__{stem}.pdf',
    )
    logger.info('Permutation plot saved to %s.', plots_dir / f'permutation__{stem}.pdf')

    grouped_perm_df = compute_grouped_permutation_importance(
        artifact, X_test, y_test, group_labels, scorer=scorer, scoring_label=scoring_label,
        greater_is_better=greater_is_better, n_repeats=n_repeats,
        out_path=results_dir / f'permutation_grouped__{stem}.csv',
    )
    plot_feature_importance(
        grouped_perm_df.assign(feature=grouped_perm_df['group'] + ' (×' + grouped_perm_df['n_features'].astype(str) + ')'),
        top_n=len(grouped_perm_df), xlabel=xlabel, title=label,
        out_path=plots_dir / f'permutation_grouped__{stem}.pdf',
    )
    logger.info('Grouped permutation plot saved to %s.', plots_dir / f'permutation_grouped__{stem}.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True,
                        help='Path to a markers/stacked artefact produced by ml.train.')
    parser.add_argument('--cpg-oof', type=Path, default=None,
                        help='Required for --features stacked: the CpG base model\'s test-set predictions '
                             '(a test_predictions__*.csv), matching what ml.evaluate uses.')
    parser.add_argument('--marker-oof', type=Path, default=None,
                        help='Required for --features stacked: the marker base model\'s test-set predictions, used as '
                             'the second view. Must match the --marker-oof passed at train time.')
    parser.add_argument('--n-repeats', type=int, default=100,
                        help='Permutation repeats per feature (default: 100).')
    parser.add_argument('--results-dir', type=Path, default=IMPORTANCE_DIR)
    parser.add_argument('--plots-dir', type=Path, default=IMPORTANCE_PLOTS)
    parser.add_argument('--label', default=None,
                        help='Model name to title the plots with. Defaults to the artefact\'s feature-source display '
                             'name (e.g. "Marker-based"), which is what distinguishes the models composited side by '
                             'side in a manuscript panel; pass an empty string for untitled plots.')
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)

    missing = {'task', 'features'} - set(artifact)
    if missing:
        raise KeyError(
            f'Artifact at {args.artifact} is missing required keys {sorted(missing)}. '
            f'Re-train with the current ml.train (which stamps task/features/...) to run importance on it.'
        )
    if artifact['features'] not in ('markers', 'stacked'):
        raise SystemExit(
            f"ml.importance only supports --features markers/stacked; got {artifact['features']!r}. "
            f"Per-CpG importance is intentionally not handled here."
        )

    task = TASKS[artifact['task']]
    feature_source = build_feature_source(
        artifact['features'], args.cpg_oof, split='test', marker_oof_path=args.marker_oof,
    )
    X_test, y_test = feature_source.load(task)

    run(
        artifact, task, X_test, y_test, stem=args.artifact.stem,
        n_repeats=args.n_repeats,
        results_dir=args.results_dir, plots_dir=args.plots_dir,
        label=args.label if args.label is not None else SOURCE_LABELS.get(artifact['features']),
    )


if __name__ == '__main__':
    main()
