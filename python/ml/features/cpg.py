"""Raw CpG beta-value feature source."""
import logging

import pandas as pd

from ...config import _cfg_path
from ..ajcc import load_ajcc_metadata
from ..tasks import Task


logger = logging.getLogger(__name__)


ENCODING = {'IM': 0, 'NIM': 1, 'NV': 2, 'other': 42}


class CpGFeatures:
    """Loads imputed beta values (samples × CpGs) and aligns them with the target column appropriate for the task.

    Classification target uses :data:`ENCODING` and drops the ``'other'`` class. Ordinal target reuses the therapeutic
    group mapping from :func:`ml.ajcc.load_ajcc_metadata`, which restricts to IM+NIM tumours with a valid AJCC stadium.

    :ivar supports_cpg_pipeline: Always ``True`` — CV loop applies the filter/reducer steps for this source.
    """

    supports_cpg_pipeline = True

    def __init__(self, split: str = 'train'):
        """:param split: Which beta value file to load (``'train'`` or ``'test'``)."""
        self.split = split

    def load(self, task: Task) -> tuple[pd.DataFrame, pd.Series]:
        """Load betas and align them to ``task.target_col``.

        :param task: Task definition.
        :return: ``(X, y)`` with ``X`` of shape (n_samples, n_cpgs) and ``y`` aligned by index.
        """
        logger.info('Loading betas_imputed_%s...', self.split)
        betas = pd.read_csv(_cfg_path(f'betas_imputed_{self.split}'), index_col=0)  # CpGs × samples
        X = betas.T

        if task.target_col == 'therapeutic_group':
            meta = load_ajcc_metadata()
            meta.index = meta.index.astype(str)
            common = X.index.intersection(meta.index)
            X = X.loc[common]
            y = meta.loc[common, 'therapeutic_group'].astype(int)
            logger.info('Loaded %d samples, %d CpGs (target=therapeutic_group).', X.shape[0], X.shape[1])
            logger.info('Therapeutic group distribution:\n%s', y.value_counts().sort_index().to_string())
            return X, y

        logger.info('Loading meta_data...')
        meta = pd.read_csv(_cfg_path('meta_data'), index_col='slideId')
        common = X.index.intersection(meta.index)
        X = X.loc[common]
        y = meta.loc[common, task.target_col].map(ENCODING)

        # Drop 'other' class (encoded as 42)
        mask = y != 42
        X = X.loc[mask]
        y = y.loc[mask]

        logger.info('Loaded %d samples, %d CpGs.', X.shape[0], X.shape[1])
        logger.info('Class distribution:\n%s', y.value_counts().to_string())
        return X, y

    def prepare_fold(self, fold, train_idx, val_idx, X, y, task):
        """Identity hook — raw CpG inputs need no per-fold rewrite."""
        return X, y
