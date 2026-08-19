"""CpG-derived marker feature source: Horvath EAA, EpiScore cell-type fractions, CNV burden.

Two of the features get a per-fold, leakage-safe treatment in :meth:`prepare_fold`:

* **CNV burden** (classification only) — the base values in ``cnv_burden_results.csv`` are swapped, for validation rows
  only, with the per-fold file ``cnv_burden_results_fold{k}.csv`` (computed against a nevus reference rebuilt from
  training-fold nevi only), so val rows never peek at their own validation reference. This matters only when a nevus can
  be a validation sample; the ordinal task (IM/NIM only) has no nevi, so it keeps the base all-nevus burden, which is
  already leak-free for every ordinal sample.
* **Horvath EAA** — the epigenetic age acceleration is the residual of regressing the skinHorvath epigenetic age on
  chronological age. The methylclock ``ageAcc2`` column fits that line across the *whole* dataset, leaking validation
  rows into their own acceleration value. ``R/analysis/horvath_eaa.R`` instead precomputes two residuals (mirroring
  the CNV base/per-fold split) per task in ``horvath_eaa__<task>.csv``: ``eaa_base``, fit on the training cohort (used
  by the final model / test set and as the training-row value in CV), and ``eaa_oof``, each sample's residual under a
  fold that excluded it. The val rows are swapped to ``eaa_oof`` per fold (see :func:`swap_val_horvath`) — train rows
  keep ``eaa_base``.

The loaders and the ``*_COLS`` constants here are the single definition of the marker set: :mod:`ml.correlation`
reuses them rather than keeping its own copies.
"""
import logging

import pandas as pd

from ...config import CNV_BURDEN_DIR, _CONFIG, _cfg_path
from ..tasks import Task


logger = logging.getLogger(__name__)


def _split_slide_ids(split: str) -> pd.Index:
    """slideIds belonging to ``split``, partitioned by the config test clinic(s).

    Mirrors the train/test partition in :func:`preprocessing._train_test_split` (test rows are the ``test_clinic``
    samples, train rows are everything else), so the markers/stacked external test set matches the CpG one.

    :param split: ``'train'`` or ``'test'``.
    :return: Index of slideIds (as strings) in that split.
    """
    meta = pd.read_csv(_cfg_path('meta_data'), index_col='slideId')
    meta.index = meta.index.astype(str)
    test_clinics = _CONFIG['test_clinic']
    if isinstance(test_clinics, str):
        test_clinics = [test_clinics]
    is_test = meta['clinic'].isin(test_clinics)
    return meta.index[is_test if split == 'test' else ~is_test]


HORVATH_COLS = ['horvath_eaa']
EPISCORE_COLS = ['Melanocyte', 'Keratinocyte', 'Stromal', 'Endothelial',
                 'Th', 'T_reg', 'T_CD8', 'Other_Lymphoid', 'Myeloid']
CNV_COLS = ['fga', 'total_burden']
MARKER_COLS = HORVATH_COLS + EPISCORE_COLS + CNV_COLS

# Display names for the figures (facet titles, heatmap ticks), spelling out the abbreviated column names and dropping
# the underscores. Columns absent here are already presentable and are shown verbatim, so this maps only the
# exceptions (applied in :func:`ml.correlation.run_all`).
MARKER_NAMES = {
    'horvath_eaa': 'Horvath EAA',
    'Th': 'T helper',
    'T_reg': 'Regulatory T',
    'T_CD8': 'CD8+ T',
    'Other_Lymphoid': 'Other lymphoid',
    'fga': 'FGA',
    'total_burden': 'Total burden',
}


def _load_horvath_eaa(task: Task) -> pd.DataFrame:
    """Fold-aware Horvath EAA precomputed by ``R/analysis/horvath_eaa.R`` for ``task``.

    The R script writes one file per task (config key ``horvath_eaa_<task>``): ``eaa_base`` is identical across tasks,
    but ``eaa_oof`` follows each task's own fold assignment, so the correct file must be selected by task.

    :param task: Task definition — selects the ``horvath_eaa_<task.name>`` config path.
    :return: DataFrame indexed by ``slideId`` with ``eaa_base`` (training-cohort fit; final model + test set) and
        ``eaa_oof`` (held-out-fold fit; swapped into CV validation rows). ``eaa_oof`` is NaN for rows not in any CV
        fold (e.g. test-clinic samples).
    """
    df = pd.read_csv(_cfg_path(f'horvath_eaa_{task.name}')).set_index('slideId')[['eaa_base', 'eaa_oof']]
    df.index = df.index.astype(str)
    return df


def swap_val_horvath(X: pd.DataFrame, val_idx, eaa: pd.DataFrame) -> pd.DataFrame:
    """Replace the val rows' Horvath EAA with the out-of-fold residual (line fit excluding those rows).

    Training rows keep ``eaa_base`` (fit on the whole training cohort) — only the val side must not see itself, exactly
    as for the CNV swap. ``eaa_oof`` is leakage-safe per sample (each value's regression was fit on a fold that
    excluded that sample), so the swap is correct regardless of how the CV loop's fold indices line up.

    :param X: Feature matrix containing the :data:`HORVATH_COLS` column; not modified (a copy is returned).
    :param val_idx: Positional indices of validation rows in ``X``.
    :param eaa: Frame from :func:`_load_horvath_eaa`.
    :return: A copy of ``X`` with val-row Horvath EAA swapped where an out-of-fold residual is available.
    """
    X = X.copy()
    val_ids = X.index[val_idx]
    oof = eaa['eaa_oof'].dropna()
    common_val = val_ids.intersection(oof.index)
    X.loc[common_val, HORVATH_COLS[0]] = oof.loc[common_val]
    missing = len(val_ids) - len(common_val)
    if missing:
        logger.warning('%d val samples missing an out-of-fold Horvath EAA — kept base residual.', missing)
    return X


def _load_episcore() -> pd.DataFrame:
    df = pd.read_csv(_cfg_path('deconv_episcore')).set_index('slideId')[EPISCORE_COLS]
    df.index = df.index.astype(str)
    return df


def _load_cnv_base() -> pd.DataFrame:
    df = pd.read_csv(CNV_BURDEN_DIR / 'cnv_burden_results.csv')
    df['slideId'] = df['slideId'].astype(str)
    return df.set_index('slideId')[CNV_COLS]


def _load_cnv_fold(fold: int) -> pd.DataFrame:
    """Per-fold CNV burden against a training-fold-only nevus reference.

    :param fold: 0-based fold index.
    :return: DataFrame indexed by ``slideId`` with the columns in :data:`CNV_COLS`.
    """
    path = CNV_BURDEN_DIR / f'cnv_burden_results_fold{fold}.csv'
    df = pd.read_csv(path)
    df['slideId'] = df['slideId'].astype(str)
    return df.set_index('slideId')[CNV_COLS]


def swap_val_cnv(X: pd.DataFrame, val_idx, fold: int) -> pd.DataFrame:
    """Replace the val rows' CNV burden columns with values from the per-fold reference.

    Training rows keep the base burden — only the val side is sensitive to leakage from the
    nevus reference used in the global file.

    :param X: Feature matrix containing :data:`CNV_COLS`; not modified (a copy is returned).
    :param val_idx: Positional indices of validation rows in ``X``.
    :param fold: 0-based fold index, used to locate ``cnv_burden_results_fold{fold}.csv``.
    :return: A copy of ``X`` with val-row CNV columns swapped where the per-fold file has them.
    """
    cnv_fold = _load_cnv_fold(fold)
    X = X.copy()
    val_ids = X.index[val_idx]
    common_val = val_ids.intersection(cnv_fold.index)
    X.loc[common_val, CNV_COLS] = cnv_fold.loc[common_val, CNV_COLS]
    missing = len(val_ids) - len(common_val)
    if missing:
        logger.warning('Fold %d: %d val samples missing from cnv_burden_results_fold%d.csv', fold, missing, fold)
    return X


def _load_metadata_target(task: Task) -> pd.Series:
    """Pull the target Series for ``task``.

    Ordinal reuses the canonical therapeutic group mapping from :func:`ml.ajcc.load_ajcc_metadata`; classification
    maps via :data:`ml.features.cpg.ENCODING`.

    :param task: Task definition.
    :return: Target Series indexed by ``slideId``.
    """
    if task.target_col == 'therapeutic_group':
        from ..ajcc import load_ajcc_metadata
        meta = load_ajcc_metadata()
        meta.index = meta.index.astype(str)
        return meta['therapeutic_group'].astype(int)

    from .cpg import ENCODING
    meta = pd.read_csv(_cfg_path('meta_data'), index_col='slideId')
    y = meta[task.target_col].map(ENCODING)
    y = y[y != 42]
    y.index = y.index.astype(str)
    return y


class MarkerFeatures:
    """Twelve CpG-derived marker features (Horvath EAA + EpiScore cells + CNV burden).

    :ivar supports_cpg_pipeline: Always ``False`` — features are already low-dim, so filter/reducer are skipped.
    """

    supports_cpg_pipeline = False

    def __init__(self, split: str = 'train'):
        """:param split: ``'train'`` or ``'test'`` — partitions samples by the config test clinic, as CpGFeatures does."""
        self.split = split

    def load(self, task: Task) -> tuple[pd.DataFrame, pd.Series]:
        """Load and align all 12 marker features against ``task.target_col``.

        Samples are restricted to :attr:`split` (train = non-test-clinic, test = test-clinic), so external-test evaluation
        scores the model on unseen samples rather than its own training rows.

        The Horvath EAA column is filled with ``eaa_base`` (residual fit on the training cohort); :meth:`prepare_fold`
        swaps the validation rows to their out-of-fold residual during CV. The base value is the right default for the
        non-CV paths (final-model training, external-test evaluation), where there are no folds.

        :param task: Task definition.
        :return: ``(X, y)`` with ``X`` having :data:`MARKER_COLS` as columns.
        """
        y = _load_metadata_target(task)
        self._eaa = _load_horvath_eaa(task)
        episcore = _load_episcore()
        cnv = _load_cnv_base()

        horvath = self._eaa['eaa_base'].rename(HORVATH_COLS[0])
        X = horvath.to_frame().join(episcore, how='outer').join(cnv, how='outer')
        common = X.index.intersection(y.index).intersection(_split_slide_ids(self.split))
        X = X.loc[common, MARKER_COLS].fillna(0)
        y = y.loc[common]
        logger.info('Loaded %d %s samples, %d marker features.', X.shape[0], self.split, X.shape[1])
        return X, y

    def prepare_fold(self, fold, train_idx, val_idx, X, y, task):
        """Apply the per-fold, leakage-safe overrides for the validation rows: CNV burden swap and Horvath EAA swap.

        Both touch validation rows only — CNV to the fold's train-only nevus reference, Horvath EAA to the out-of-fold
        residual. Training rows keep their base values. The CNV swap runs for classification only: it guards against a
        validation nevus landing in its own nevus reference, and the ordinal task (IM/NIM only) has no nevi, so it keeps
        the base all-nevus burden — already leak-free for every ordinal sample.

        :param fold: 0-based fold index.
        :param train_idx: Positional indices of training rows in ``X`` (unused; kept for protocol compatibility).
        :param val_idx: Positional indices of validation rows in ``X``.
        :param X: Feature matrix to rewrite (a copy is made internally).
        :param y: Target series, returned unchanged.
        :param task: Task definition — gates the CNV swap (classification only).
        :return: ``(X_with_per_fold_markers, y)``.
        """
        if task.name == 'classification':
            X = swap_val_cnv(X, val_idx, fold)
        X = swap_val_horvath(X, val_idx, self._eaa)
        return X, y
