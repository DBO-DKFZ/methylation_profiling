"""Unified cross-validation loop for any (Task, FeatureSource) combination.

The body has no ``task_type`` branching — everything task-specific is read off the :class:`Task` instance:

* metric computation (``task.metric_fns``),
* OOF schema (``task.oof_schema``),
* sklearn predict method (``task.predict_method``).

Ordinal models are scored on their ``predict`` output; those that also expose ``predict_proba`` (the ordinal
classifiers) get ``prob_<class>`` columns written alongside the ``prediction`` column for downstream stacking.

Filter/reducer are applied only when the feature source is raw CpGs (``feature_source.supports_cpg_pipeline``).
"""
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CV_DIR, RANDOM_STATE
from .features.base import FeatureSource
from .tasks import Task

logger = logging.getLogger(__name__)


def _derive_groups(slide_ids: pd.Index) -> pd.Series:
    """Derive patient identifiers from slide IDs by stripping the trailing ``-1`` or ``-2`` suffix.

    Used both to construct groups for ``StratifiedGroupKFold`` (so all slides from the same patient land in the same
    fold) and to cluster the patient-level bootstrap (so a patient's slides are resampled together, not as
    independent observations).

    :param slide_ids: Index of slide IDs (e.g. ``'ABC123-1'``).
    :return: Series of patient IDs aligned to ``slide_ids``.
    """
    return slide_ids.to_series().str.replace(r'-[12]$', '', regex=True)


def _supervised_filter_kwargs(filter_name: str, task: Task) -> dict:
    """Forward ``task=`` to filters that depend on ``y`` semantically.

    :param filter_name: Filter registry key.
    :param task: Task definition (only its ``.name`` is forwarded).
    :return: Kwargs dict to merge into the filter call, empty if the filter is not in ``SUPERVISED_FILTERS``.
    """
    from .filters import SUPERVISED_FILTERS
    if filter_name in SUPERVISED_FILTERS:
        return {'task': task.name}
    return {}


def _reducer_kwargs(reducer_cls, reducer_kwargs: dict | None, task: Task) -> dict:
    """Inject ``task=`` into the kwargs of task-aware reducers (currently only :class:`PLSReducer`).

    :param reducer_cls: Reducer class to instantiate.
    :param reducer_kwargs: Existing kwargs from the registry, or ``None``.
    :param task: Task definition (only its ``.name`` is forwarded).
    :return: Merged kwargs dict.
    """
    from .reducers import PLSReducer
    base = dict(reducer_kwargs or {})
    if reducer_cls is PLSReducer and 'task' not in base:
        base['task'] = task.name
    return base


def _append_record(path: Path, record: dict) -> None:
    """Append a single per-fold record to the checkpoint CSV, writing header on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, mode='a', header=not path.exists(), index=False)


def build_predictions_df(
    index: pd.Index,
    y_true: pd.Series,
    task: Task,
    *,
    classes: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    include_fold: bool = True,
) -> pd.DataFrame:
    """Build a per-sample prediction DataFrame matching ``task.oof_schema``.

    Used both to allocate an empty OOF frame (CV loop fills it fold-by-fold) and to materialise a fully-populated
    prediction frame for external-test evaluation (single call with ``y_pred`` / ``y_prob``).

    For the ``'prediction'`` schema, ``prob_<class>`` columns are added alongside ``prediction`` when ``classes`` is
    given (i.e. the model exposed ``predict_proba``); metrics still use ``prediction``, the probabilities are extra
    data for downstream stacking.

    :param index: Row index (typically ``X.index``).
    :param y_true: Ground-truth target series; written into the ``y_true`` column.
    :param task: Task definition.
    :param classes: The class labels that determine ``prob_<class>`` columns. Required to write probabilities in either
        schema.
    :param y_pred: Optional predictions; populates the ``prediction`` column for ``oof_schema='prediction'``.
    :param y_prob: Optional ``(n_samples, n_classes)`` probability matrix; populates ``prob_<class>`` columns.
    :param include_fold: Whether to allocate the ``fold`` column (left NaN). The CV loop needs it; the evaluation path
        does not.
    :return: DataFrame indexed by ``index`` with task-appropriate prediction columns and a populated ``y_true`` column.
    """
    trailing = ['y_true', 'fold'] if include_fold else ['y_true']
    prob_cols = [f'prob_{int(c)}' for c in classes] if classes is not None else []
    if task.oof_schema == 'prediction':
        df = pd.DataFrame(index=index, columns=['prediction', *prob_cols, *trailing], dtype=float)
        if y_pred is not None:
            df['prediction'] = np.asarray(y_pred)
    else:
        df = pd.DataFrame(index=index, columns=[*prob_cols, *trailing], dtype=float)
    if y_prob is not None and classes is not None:
        for i, c in enumerate(classes):
            df[f'prob_{int(c)}'] = y_prob[:, i]
    df['y_true'] = y_true
    return df


def run_cv(
    task: Task,
    feature_source: FeatureSource,
    X: pd.DataFrame,
    y: pd.Series,
    models: dict,
    n_splits: int = 5,
    upsample: bool = True,
    filter_fn: Optional[Callable] = None,
    filter_kwargs: dict | None = None,
    filter_name: str = 'none',
    reducers: dict | None = None,
    collect_oof: bool = False,
    skip_combos: set[tuple[str, str, str]] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Run patient-stratified k-fold CV for the given task and feature source.

    For raw-CpG feature sources, the CpG filter is applied once per fold (on training data only) and shared across
    reducers/models. For low-dim feature sources (markers/stacked) the filter/reducer steps are skipped: those
    features are already low-dimensional and meaningfully named.

    :param task: Task definition (classification or ordinal).
    :param feature_source: Provides ``prepare_fold`` (per-fold CNV swap) and ``supports_cpg_pipeline``. Load ``X, y``
        via ``feature_source.load(task)`` in the caller so they can be reused across multiple ``run_cv`` calls (e.g.
        across filters in a benchmark).
    :param X: Feature matrix (rows = slides, index = slide IDs).
    :param y: Target series, aligned with ``X``.
    :param models: ``{name: sklearn estimator}`` — cloned per fold.
    :param n_splits: Number of CV folds.
    :param upsample: SMOTE upsampling of the training fold (treats integer y values as classes).
    :param filter_fn: CpG-mask filter ``f(betas, y, **kwargs) -> bool Series``. Ignored for low-dim feature sources.
    :param filter_kwargs: Extra kwargs to forward to ``filter_fn``.
    :param filter_name: Label used in the output DataFrame's ``filter`` column.
    :param reducers: ``{name: (reducer_cls_or_None, kwargs_or_None)}``. Ignored for low-dim feature sources (substituted
        with ``{'none': (None, None)}``).
    :param collect_oof: If True, return ``(metrics_df, oof_store)``; ``oof_store`` is keyed by
        ``(reducer_name, model_name)`` and holds per-task OOF DataFrames.
    :param skip_combos: ``(filter, reducer, model)`` triples to skip — used to resume after a crash by skipping combos
        already present in the checkpoint.
    :param checkpoint_path: If set, each per-fold record is appended to this CSV as it's computed, so partial
        progress survives a crash.
    """
    filter_kwargs = dict(filter_kwargs or {})
    skip_combos = set(skip_combos or set())
    if feature_source.supports_cpg_pipeline:
        _reducers = reducers if reducers is not None else {'none': (None, None)}
    else:
        _reducers = {'none': (None, None)}
        filter_fn = None
        filter_name = 'none'

    groups = _derive_groups(X.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    records: list[dict] = []
    oof_store: dict = {}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        logger.info(
            'Fold %d/%d — task=%s, feature_source=%s, filter=%s, reducers=%s, training %d models...',
            fold + 1, n_splits, task.name, feature_source.__class__.__name__,
            filter_name, list(_reducers), len(models),
        )

        # Per-fold rewrite (CNV swap for markers/stacked; identity for CpGs).
        X_fold, y_fold = feature_source.prepare_fold(fold, train_idx, val_idx, X, y, task)

        X_train, X_val = X_fold.iloc[train_idx], X_fold.iloc[val_idx]
        y_train, y_val = y_fold.iloc[train_idx], y_fold.iloc[val_idx]

        # CpG filtering on the training fold only.
        if filter_fn is None:
            selected = X_train.columns
        else:
            kw = {**filter_kwargs, **_supervised_filter_kwargs(filter_name, task)}
            cpg_mask = filter_fn(X_train.T, y=y_train, **kw)
            selected = cpg_mask[cpg_mask].index
            if len(selected) == 0:
                logger.warning('Fold %d: filter "%s" selected 0 CpGs — skipping fold.', fold + 1, filter_name)
                continue
        X_train_f = X_train[selected]
        X_val_f = X_val[selected]

        for reducer_name, (reducer_cls, reducer_kwargs) in _reducers.items():
            active_models = {n: m for n, m in models.items()
                             if (filter_name, reducer_name, n) not in skip_combos}
            if not active_models:
                logger.info('  Skipping reducer=%s (all models done).', reducer_name)
                continue
            if reducer_cls is not None:
                reducer = reducer_cls(**_reducer_kwargs(reducer_cls, reducer_kwargs, task))
                X_tr = reducer.fit_transform(X_train_f, y_train)
                X_va = reducer.transform(X_val_f)
            else:
                X_tr, X_va = X_train_f, X_val_f

            # SMOTE per reducer-specific training data. ``k_neighbors`` adapts to rare classes
            # (rare therapeutic groups can drop below the default of 5 samples per fold).
            if upsample:
                cols = X_tr.columns if isinstance(X_tr, pd.DataFrame) else None
                k = min(5, int(y_train.value_counts().min()) - 1)
                X_tr, y_tr = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X_tr, y_train)
                if cols is not None:
                    X_tr = pd.DataFrame(X_tr, columns=cols)
            else:
                y_tr = y_train

            for model_name, model in active_models.items():
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)

                if task.oof_schema == 'probs':
                    y_prob = fold_model.predict_proba(X_va)
                    y_pred = fold_model.classes_[np.argmax(y_prob, axis=1)]
                else:
                    # Ordinal: score on the model's own predict; keep probabilities (when available) as extra
                    # columns for stacking without letting them change the metric.
                    y_pred = fold_model.predict(X_va)
                    y_prob = fold_model.predict_proba(X_va) if hasattr(fold_model, 'predict_proba') else None

                # Compute metrics from the task registry — one body for both tasks.
                record = {
                    'filter':   filter_name,
                    'reducer':  reducer_name,
                    'model':    model_name,
                    'fold':     fold,
                    'n_features': X_tr.shape[1],
                }
                for mname, mfn in task.metric_fns.items():
                    record[mname] = mfn(y_val, y_pred, y_prob)
                records.append(record)
                if checkpoint_path is not None:
                    _append_record(checkpoint_path, record)

                headline = ' | '.join(
                    f'{m}={record[m]:.4f}' for m in task.headline_metrics if m in record and pd.notna(record[m])
                )
                logger.info('  reducer=%-12s | model=%-15s | %s', reducer_name, model_name, headline)

                if collect_oof:
                    key = (reducer_name, model_name)
                    if key not in oof_store:
                        classes = getattr(fold_model, 'classes_', None) if y_prob is not None else None
                        oof_store[key] = build_predictions_df(X.index, y, task, classes=classes)
                    df_oof = oof_store[key]
                    if task.oof_schema != 'probs':
                        df_oof.loc[X_val.index, 'prediction'] = y_pred
                    if y_prob is not None:
                        for i, cls in enumerate(fold_model.classes_):
                            df_oof.loc[X_val.index, f'prob_{int(cls)}'] = y_prob[:, i]
                    df_oof.loc[X_val.index, 'fold'] = fold

    metrics_df = pd.DataFrame(records)
    if collect_oof:
        return metrics_df, oof_store
    return metrics_df


def summarize(results: pd.DataFrame, task: Task, out_name: str = 'cv_results.csv') -> None:
    """Log mean ± std per ``(filter, reducer, model)`` and persist per-fold results.

    Output goes to ``CV_DIR/<out_name>``. The summary lists headline metrics first, then any
    ``mae_group_*`` per-group details when present.

    :param results: Per-fold metrics DataFrame as produced by :func:`run_cv`.
    :param task: Task definition (drives the headline-metric ordering).
    :param out_name: Filename written under ``CV_DIR``.
    """
    headline = list(task.headline_metrics)
    per_group_cols = [c for c in results.columns if c.startswith('mae_group_')]
    metric_cols = headline + per_group_cols

    grouped = results.groupby(['filter', 'reducer', 'model'])[metric_cols]
    means, stds = grouped.mean(), grouped.std()

    logger.info('=== SUMMARY (task=%s) ===', task.name)
    for (filter_name, reducer_name, model_name), row in means.iterrows():
        s = stds.loc[(filter_name, reducer_name, model_name)]
        parts = [f'filter={filter_name}', f'reducer={reducer_name}', f'model={model_name:<15s}']
        for m in headline:
            parts.append(f'{m}={row[m]:.3f}±{s[m]:.3f}')
        logger.info(' | '.join(parts))
        if per_group_cols:
            group_parts = [f'group{c.rsplit("_", 1)[-1]}={row[c]:.2f}' for c in per_group_cols if pd.notna(row[c])]
            if group_parts:
                logger.info('    per-group MAE: %s', ' | '.join(group_parts))

    CV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CV_DIR / out_name
    results.to_csv(out_path, index=False)
    logger.info('Full per-fold results saved to %s', out_path)


