"""Stacked feature source: CpG-model predictions joined with marker-model predictions.

Both prediction files are constructor arguments, and the schema (``prob_*`` columns vs a single ``prediction``
column) is detected from the file itself, so stacking works for both the classification and the ordinal pipeline.

The CpG prediction columns are joined with a *second* prediction CSV, the marker classifier's own out-of-fold
(train) / test predictions. This is the two-base-learner stacking design: both views are on the same probability
scale and the meta-feature space is small (2×n_classes). Both prediction sets are already leakage-safe by
construction (produced by :mod:`ml.oof` under the same fold structure), so no per-fold swap is applied.
"""
import logging
from pathlib import Path

import pandas as pd

from ..tasks import Task
from .markers import _load_metadata_target


logger = logging.getLogger(__name__)


class StackedFeatures:
    """Concatenate a CpG prediction CSV's columns with a marker prediction CSV's columns.

    :ivar supports_cpg_pipeline: Always ``False`` — features are already low-dim.
    """

    supports_cpg_pipeline = False

    def __init__(self, cpg_oof_path: Path | str, marker_oof_path: Path | str):
        """:param cpg_oof_path: Path to the CpG prediction CSV — the training OOF from :mod:`ml.oof` (CV/train) or the
            base model's test predictions (eval), indexed by ``slideId``.
        :param marker_oof_path: Path to the marker classifier's prediction CSV (the second view). Must match
            ``cpg_oof_path``'s split (both train OOF or both test predictions).
        """
        self.cpg_oof_path = Path(cpg_oof_path)
        self.marker_oof_path = Path(marker_oof_path)

    @staticmethod
    def _read_oof(path: Path) -> tuple[pd.DataFrame, list[str]]:
        """:return: ``(df, pred_cols)``; ``pred_cols`` is the list of ``prob_*`` columns or ``['prediction']``."""
        df = pd.read_csv(path, index_col='slideId')
        df.index = df.index.astype(str)
        pred_cols = [c for c in df.columns if c.startswith('prob_')] or ['prediction']
        return df, pred_cols

    def load(self, task: Task) -> tuple[pd.DataFrame, pd.Series]:
        """Concatenate the CpG prediction columns with the marker prediction columns on a shared ``slideId`` index.

        :param task: Task definition (selects the metadata target).
        :return: ``(X, y)`` where ``X`` is the ``cpg_``-prefixed CpG prediction columns followed by the
            ``mrk_``-prefixed marker prediction columns.
        """
        cpg, cpg_cols = self._read_oof(self.cpg_oof_path)
        mrk, mrk_cols = self._read_oof(self.marker_oof_path)
        # Prefix both prediction blocks so the shared ``prob_*`` / ``prediction`` names don't collide on join.
        cpg = cpg[cpg_cols].add_prefix('cpg_')
        mrk = mrk[mrk_cols].add_prefix('mrk_')
        y = _load_metadata_target(task)
        common = cpg.index.intersection(mrk.index).intersection(y.index)
        X = cpg.loc[common].join(mrk.loc[common], how='inner')
        y = y.loc[X.index]
        self._pred_cols = list(X.columns)
        logger.info('Loaded %d samples, %d features (%d CpG-pred + %d marker-pred).',
                    X.shape[0], X.shape[1], len(cpg_cols), len(mrk_cols))
        return X, y

    def prepare_fold(self, fold, train_idx, val_idx, X, y, task):
        """Identity — both prediction views are already leakage-safe (produced by :mod:`ml.oof` under the same fold
        structure), so there is nothing to swap per fold. Kept for :class:`FeatureSource` protocol conformance.

        :return: ``(X, y)`` unchanged.
        """
        return X, y
