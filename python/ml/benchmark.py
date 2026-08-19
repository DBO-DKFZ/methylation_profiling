"""Unified benchmark CLI covering the (task × feature-source) matrix.

* ``--task {classification,ordinal}`` picks the target.
* ``--features {cpg,markers,stacked}`` picks the input feature space.
* ``--cpg-oof <path>`` is required when ``--features stacked``.

Filter/reducer flags are honoured only for ``--features cpg``; the other feature sources are already low-dimensional and
named, and skip those steps.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import CV_DIR, RANDOM_STATE
from ._registries import FILTERS, REDUCERS
from .cv import run_cv, summarize
from .features import build as build_feature_source
from .models import MODELS
from .tasks import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def _select(d: dict, names: list[str] | None) -> dict:
    if not names:
        return d
    missing = [n for n in names if n not in d]
    if missing:
        raise ValueError(f'Unknown name(s): {missing}. Available: {list(d)}')
    return {n: d[n] for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), required=True,
                        help='Prediction target: classification (IM/NIM/NV) or ordinal (therapeutic groups 0-5).')
    parser.add_argument('--features', choices=['cpg', 'markers', 'stacked'], required=True,
                        help='Feature source.')
    parser.add_argument('--cpg-oof', type=Path, default=None,
                        help='Path to a CpG OOF CSV. Required for --features stacked.')
    parser.add_argument('--marker-oof', type=Path, default=None,
                        help='Path to a marker OOF CSV. Required for --features stacked: the marker classifier '
                             'predictions are the second view (two-base-learner stacking).')
    parser.add_argument('--filters', nargs='*', default=None,
                        help=f'Subset of {list(FILTERS)} (default: all). Ignored unless --features cpg.')
    parser.add_argument('--reducers', nargs='*', default=None,
                        help=f'Subset of {list(REDUCERS)} (default: all). Ignored unless --features cpg.')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Subset of the model registry for the chosen task. Default: all.')
    parser.add_argument('--no-upsample', action='store_true',
                        help='Disable SMOTE upsampling on the training fold.')
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    task = TASKS[args.task]
    feature_source = build_feature_source(args.features, args.cpg_oof, marker_oof_path=args.marker_oof)
    feature_kind = 'cpg' if args.features == 'cpg' else 'lowdim'
    models = _select(MODELS[task.name][feature_kind], args.models)

    if args.features == 'cpg':
        active_filters = _select(FILTERS, args.filters)
        active_reducers = _select(REDUCERS, args.reducers)
    else:
        active_filters = {'none': None}
        active_reducers = {'none': (None, None)}

    out_name = f'cv_results__{task.name}__{args.features}.csv'
    ckpt_path = CV_DIR / f'cv_results__{task.name}__{args.features}.checkpoint.csv'

    # Resume: skip (filter, reducer, model) combos that already have all folds in the checkpoint.
    n_splits = 5
    skip_combos: set[tuple[str, str, str]] = set()
    if ckpt_path.exists():
        prior = pd.read_csv(ckpt_path)
        if not prior.empty:
            counts = prior.groupby(['filter', 'reducer', 'model'])['fold'].nunique()
            done = set(counts[counts >= n_splits].index)
            skip_combos |= done
            logger.info('Checkpoint at %s contains %d fully-completed combos; resuming.', ckpt_path, len(done))

    # Load X, y once and reuse across all filter iterations — the filter mask is the only thing that varies here, so
    # there's no point paying the (potentially large) load cost per filter.
    X, y = feature_source.load(task)

    for filter_name, filter_fn in active_filters.items():
        logger.info('Running CV: task=%s, features=%s, filter=%s, reducers=%s, models=%s',
                    task.name, args.features, filter_name, list(active_reducers), list(models))
        run_cv(
            task=task,
            feature_source=feature_source,
            X=X,
            y=y,
            models=models,
            n_splits=n_splits,
            reducers=active_reducers,
            filter_fn=filter_fn,
            filter_name=filter_name,
            upsample=not args.no_upsample,
            skip_combos=skip_combos,
            checkpoint_path=ckpt_path,
        )

    # The checkpoint is the source of truth — it contains every per-fold record from this and prior runs.
    final = pd.read_csv(ckpt_path) if ckpt_path.exists() else pd.DataFrame()
    summarize(final, task=task, out_name=out_name)


if __name__ == '__main__':
    main()
