"""Export the patient-stratified k-fold split to ``cv_folds__<task>.csv``.

The filename carries the task because classification and ordinal use different targets (and different sample sets)
and therefore land samples in different folds. The R-side per-fold CNV burden pipeline (and Python's
:class:`ml.features.markers.MarkerFeatures`) read this file to align with the splits used internally by
:func:`ml.cv.run_cv`.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CV_DIR, RANDOM_STATE
from .cv import _derive_groups
from .features import CpGFeatures
from .tasks import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def export_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Materialise the same splits :func:`ml.cv.run_cv` uses for a task.

    Stratifies on the raw target ``y`` (as the CV loop does) so the exported folds match it bit-for-bit.

    :param X: Feature matrix indexed by ``slideId``.
    :param y: Target aligned to X.
    :param n_splits: Number of CV folds.
    :param out_path: If given, write the assignments as CSV.
    :return: DataFrame indexed by slide ID with a single ``val_fold`` column.
    """
    groups = _derive_groups(X.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    val_fold = pd.Series(-1, index=X.index, name='val_fold', dtype=int)
    for fold, (_, val_idx) in enumerate(cv.split(X, y, groups)):
        val_fold.iloc[val_idx] = fold

    assert (val_fold >= 0).all(), 'every sample must land in exactly one validation fold'

    out = val_fold.to_frame()
    logger.info('Fold sizes (val):\n%s', out['val_fold'].value_counts().sort_index().to_string())

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index_label='slideId')
        logger.info('Fold assignments saved to %s.', out_path)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default='classification')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV path (default: <CV_DIR>/cv_folds__<task>.csv).')
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    task = TASKS[args.task]
    X, y = CpGFeatures().load(task)
    export_folds(X, y, out_path=args.out or (CV_DIR / f'cv_folds__{task.name}.csv'))


if __name__ == '__main__':
    main()
