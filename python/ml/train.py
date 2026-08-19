"""Train a single (filter, reducer, model) pipeline on the full dataset and persist it.

Works for both tasks (classification, ordinal) and both raw-CpG and low-dim feature sources — the artifact records the
filter mask, fitted reducer (if any), and trained estimator.
"""
import argparse
import logging
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator

from ..config import MODELS_DIR, RANDOM_STATE
from ._registries import FILTERS, REDUCERS, resolve_pipeline
from .features import build as build_feature_source
from .tasks import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    filter_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    reducer: Optional[tuple] = None,
    upsample: bool = True,
    out_path: Optional[Path] = None,
) -> dict:
    """
    Trains a model on the full dataset with optional CpG filtering, dimensionality reduction, and SMOTE upsampling.
    Follows the same pipeline order as :func:`ml.cv.run_cv`: filter → reduce → upsample → train.

    :param X: Feature matrix of shape (n_samples, n_features).
    :param y: Target aligned to X.
    :param model: Scikit-learn compatible estimator instance.
    :param filter_fn: Callable taking ``(betas, y=y, **kwargs)`` and returning a boolean CpG mask, or None to use all CpGs.
    :param filter_kwargs: Additional keyword arguments forwarded to the filter function.
    :param reducer: Tuple of ``(reducer_cls, kwargs)``, or None for no reduction.
    :param upsample: Whether to apply SMOTE oversampling (default: True).
    :param out_path: If given, persist the artifact dict to this path via joblib.
    :return: Dict with ``'model'`` (fitted estimator), ``'selected_cpgs'`` (Index of CpG names), and ``'reducer'``
        (fitted instance or None).
    """
    filter_kwargs = filter_kwargs or {}

    # CpG filter
    if filter_fn is not None:
        logger.info('Applying filter to full training set...')
        cpg_mask = filter_fn(X.T, y=y, **filter_kwargs)
        selected_cpgs = cpg_mask[cpg_mask].index
        logger.info('Selected %d CpGs.', len(selected_cpgs))
        X = X[selected_cpgs]
    else:
        selected_cpgs = X.columns

    # Dimensionality reduction
    fitted_reducer = None
    if reducer is not None:
        reducer_cls, reducer_kwargs = reducer
        fitted_reducer = reducer_cls(**(reducer_kwargs or {}))
        logger.info('Applying %s reducer...', reducer_cls.__name__)
        X = fitted_reducer.fit_transform(X, y)
        logger.info('Post-reduction shape: %s', X.shape)

    # SMOTE upsampling. ``k_neighbors`` adapts to rare classes (rare therapeutic groups can drop below the default of 5).
    if upsample:
        logger.info('Applying SMOTE...')
        _cols = X.columns if isinstance(X, pd.DataFrame) else None
        k = min(5, int(y.value_counts().min()) - 1)
        X, y = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X, y)
        if _cols is not None:
            X = pd.DataFrame(X, columns=_cols)
        logger.info('Post-SMOTE shape: %s, class distribution: %s',
                    X.shape, dict(zip(*np.unique(y, return_counts=True))))

    logger.info('Training %s...', type(model).__name__)
    model.fit(X, y)
    logger.info('Training complete.')

    artifact = {'model': model, 'selected_cpgs': selected_cpgs, 'reducer': fitted_reducer}

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out_path)
        logger.info('Model saved to %s.', out_path)

    return artifact


def _default_out_path(task: str, features: str, filter_name: str, reducer_name: str, model_name: str) -> Path:
    """Stable artifact filename encoding the (task, features, filter, reducer, model) tuple."""
    return MODELS_DIR / f'model__{task}__{features}__{filter_name}__{reducer_name}__{model_name}.pkl'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), required=True)
    parser.add_argument('--features', choices=['cpg', 'markers', 'stacked'], required=True)
    parser.add_argument('--cpg-oof', type=Path, default=None,
                        help='Required when --features stacked.')
    parser.add_argument('--marker-oof', type=Path, default=None,
                        help='Required for --features stacked: the marker classifier predictions are the second '
                             'view (two-base-learner stacking).')
    parser.add_argument('--filter', default='none',
                        help=f'CpG filter name (default: none). One of: {list(FILTERS)} or "none". '
                             f'Ignored unless --features cpg.')
    parser.add_argument('--reducer', default='none',
                        help=f'Reducer name (default: none). One of: {list(REDUCERS)}. '
                             f'Ignored unless --features cpg.')
    parser.add_argument('--model', required=True,
                        help='Model name from the task/feature-specific registry in ml.models.')
    parser.add_argument('--no-upsample', action='store_true',
                        help='Disable SMOTE upsampling on the training set.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output artefact path (defaults to a name derived from the CLI args).')
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    task = TASKS[args.task]
    feature_source = build_feature_source(args.features, args.cpg_oof, marker_oof_path=args.marker_oof)
    model, filter_fn, filter_name, reducer, reducer_name = resolve_pipeline(
        task.name, args.features, args.model, args.filter, args.reducer,
        feature_source.supports_cpg_pipeline,
    )

    X, y = feature_source.load(task)
    out_path = args.out or _default_out_path(task.name, args.features, filter_name, reducer_name, args.model)

    artifact = train(
        X, y,
        model=model,
        filter_fn=filter_fn,
        reducer=reducer if reducer[0] is not None else None,
        upsample=not args.no_upsample,
        out_path=None,  # persisted below, after the artifact dict is enriched
    )

    # Stamp the pipeline identity onto the artifact so evaluate.py can rebuild the feature source and pick the right
    # task. (Stacked eval additionally needs --cpg-oof pointed at the base model's test predictions — the training OOF
    # path is not stored, as it has no test rows and is useless at eval.) Distinct keys from ``model``/``reducer``
    # (which hold the fitted estimator / reducer instance, not the registry name).
    artifact.update({
        'task':         task.name,
        'features':     args.features,
        'filter_name':  filter_name,
        'reducer_name': reducer_name,
        'model_name':   args.model,
        'upsample':     not args.no_upsample,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    logger.info('Model saved to %s.', out_path)


if __name__ == '__main__':
    main()
