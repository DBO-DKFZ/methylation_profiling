"""Generate out-of-fold predictions for a single (filter, reducer, model) combination.

The OOF schema (``prob_*`` columns vs single ``prediction``) follows the task and is consumed downstream by
:class:`ml.features.stacked.StackedFeatures` for meta-stacking and by :mod:`ml.complementarity` for the CpG↔markers
complementarity analysis.
"""
import argparse
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..config import CV_DIR, RANDOM_STATE
from ._registries import FILTERS, REDUCERS, resolve_pipeline
from .cv import run_cv
from .features import build as build_feature_source
from .tasks import TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_oof(
    task: Task,
    feature_source,
    model: BaseEstimator,
    model_name: str,
    filter_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    filter_name: str = 'none',
    reducer: Optional[tuple] = None,
    reducer_name: str = 'none',
    n_splits: int = 5,
    upsample: bool = True,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run patient-stratified k-fold CV for a single (filter, reducer, model) combination and
    return out-of-fold validation predictions. Splits and pipeline match :func:`ml.cv.run_cv`.

    :param task: Task definition.
    :param feature_source: Loader of (X, y); typically :class:`CpGFeatures` for OOF generation that downstream stacking
        will consume.
    :param model: Scikit-learn compatible estimator (cloned per fold).
    :param model_name: Used in the OOF store key and CSV filename.
    :param filter_fn: CpG mask filter (raw-CpG feature sources only) or ``None``.
    :param filter_kwargs: Extra filter kwargs.
    :param filter_name: Label.
    :param reducer: ``(reducer_cls, kwargs)`` or ``None``.
    :param reducer_name: Label.
    :param n_splits: CV folds.
    :param upsample: SMOTE on train fold.
    :param out_path: If given, write the OOF DataFrame here (CSV, indexed by ``slideId``).
    :return: OOF DataFrame.
    """
    X, y = feature_source.load(task)
    _, oof_store = run_cv(
        task=task,
        feature_source=feature_source,
        X=X,
        y=y,
        models={model_name: model},
        n_splits=n_splits,
        upsample=upsample,
        filter_fn=filter_fn,
        filter_kwargs=filter_kwargs,
        filter_name=filter_name,
        reducers={reducer_name: reducer if reducer is not None else (None, None)},
        collect_oof=True,
    )
    oof_df = oof_store[(reducer_name, model_name)]

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(out_path, index_label='slideId')
        logger.info('OOF predictions saved to %s.', out_path)

    return oof_df


def _default_out_path(task: str, features: str, filter_name: str, reducer_name: str, model_name: str) -> Path:
    """Stable OOF filename encoding the (task, features, filter, reducer, model) tuple."""
    return CV_DIR / f'oof_predictions__{task}__{features}__{filter_name}__{reducer_name}__{model_name}.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default='classification')
    parser.add_argument('--features', choices=['cpg', 'markers', 'stacked'], default='cpg')
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
                        help='Disable SMOTE upsampling on the training fold.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV path (defaults to a name derived from the CLI args).')
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    task = TASKS[args.task]
    feature_source = build_feature_source(args.features, args.cpg_oof, marker_oof_path=args.marker_oof)
    model, filter_fn, filter_name, reducer, reducer_name = resolve_pipeline(
        task.name, args.features, args.model, args.filter, args.reducer,
        feature_source.supports_cpg_pipeline,
    )

    out_path = args.out or _default_out_path(task.name, args.features, filter_name, reducer_name, args.model)

    generate_oof(
        task=task,
        feature_source=feature_source,
        model=model,
        model_name=args.model,
        filter_fn=filter_fn,
        filter_name=filter_name,
        reducer=reducer if reducer[0] is not None else None,
        reducer_name=reducer_name,
        upsample=not args.no_upsample,
        out_path=out_path,
    )


if __name__ == '__main__':
    main()
