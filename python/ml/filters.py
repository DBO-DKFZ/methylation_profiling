import logging
from typing import Literal

import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression

from ..config import RANDOM_STATE, _cfg_path
from ._utils import load_chrom_hmm_lookup

logger = logging.getLogger(__name__)


# Filters that take y semantically into account — they need ``task`` forwarded by the CV loop.
SUPERVISED_FILTERS = {'lasso', 'boruta', 'elasticnet', 'differential_methylation'}


def _variance_top_k_mask(betas: pd.DataFrame, top_k: int) -> pd.Series:
    """Return a boolean mask over ``betas.index`` selecting the top-``top_k`` highest-variance probes."""
    variances = betas.var(axis=1)
    return variances >= variances.nlargest(top_k).min()


def _elasticnet_filter(
    betas: pd.DataFrame,
    y: pd.Series,
    C: float = 1.,
    l1_ratio: float = 0.5,
    pre_filter_top_k: int | None = 50_000,
    task: Literal['classification', 'ordinal'] = 'classification',
) -> pd.Series:
    """
    Selects CpGs using an ElasticNet-regularised model.

    For classification, uses sklearn 1.8+ LogisticRegression where l1_ratio controls the penalty type
    (1.0 = L1/LASSO, 0.0 = L2/Ridge, between 0 and 1 = ElasticNet).
    For ordinal regression, swaps in :class:`sklearn.linear_model.ElasticNet` (or :class:`Lasso` when
    ``l1_ratio == 1.0``) and treats y as continuous. ``C`` is ignored — the regressors use ``alpha=0.01``
    since the two loss functions are on different scales.

    Optionally applies a variance pre-filter to reduce the feature space before fitting the regularised model.

    :param betas: DataFrame of shape (n_cpgs, n_samples).
    :param y: Target aligned to the columns of ``betas`` (integer class labels or ordinal values).
    :param C: Inverse regularisation strength (default: 1.). Ignored for ordinal.
    :param l1_ratio: ElasticNet mixing — 1.0 = LASSO, 0.0 = Ridge (default: 0.5).
    :param pre_filter_top_k: If set, restrict to the top-k highest-variance CpGs before fitting the regularised model.
        Unselected CpGs are set to False in the returned mask. Set to None to disable (default: 50_000).
    :param task: ``'classification'`` (default) or ``'ordinal'``.
    :return: Boolean Series indexed by CpG, True for selected features.
    """
    full_index = betas.index

    if pre_filter_top_k is not None and pre_filter_top_k < len(betas):
        pre_mask = _variance_top_k_mask(betas, pre_filter_top_k)
        logger.info(
            "_elasticnet_filter: variance pre-filter retained %d / %d probes (top_k=%d)",
            pre_mask.sum(), len(pre_mask), pre_filter_top_k,
        )
        betas = betas.loc[pre_mask]

    X = betas.T
    if task == 'ordinal':
        # 1/C doesn't map across loss functions — sklearn's Lasso/ElasticNet loss
        # ((1/2n)·‖y-Xw‖² + α·‖w‖₁) has a totally different scale than logistic regression.
        # α=0.1 (as in the AJCC predictor) leaves only a handful of nonzero coefs — fine for
        # prediction but too sparse for filtering; drop to 0.01 and accept every nonzero coefficient
        # (SelectFromModel's default `mean` threshold compounds the sparsity).
        alpha = 0.01
        if l1_ratio == 1.0:
            estimator = Lasso(alpha=alpha, max_iter=1000, random_state=RANDOM_STATE)
        else:
            estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=RANDOM_STATE)
        sel = SelectFromModel(estimator, threshold=1e-10)
    else:
        estimator = LogisticRegression(
            l1_ratio=l1_ratio, C=C, solver='saga', max_iter=1000,
            random_state=RANDOM_STATE,
        )
        sel = SelectFromModel(estimator)
    sel.fit(X, y)
    inner_mask = pd.Series(sel.get_support(), index=betas.index)

    # Expand back to the full feature space
    full_mask = pd.Series(False, index=full_index)
    full_mask.loc[inner_mask[inner_mask].index] = True

    logger.info(
        "_elasticnet_filter: %d / %d probes retained (task=%s, C=%s, l1_ratio=%s, pre_filter_top_k=%s)",
        full_mask.sum(), len(full_mask), task, C, l1_ratio, pre_filter_top_k,
    )
    return full_mask


def _variance_filter(betas: pd.DataFrame, y: pd.Series, top_k: int = 5000) -> pd.Series:
    """
    Selects the top-k most variable CpGs by variance across samples.

    :param betas: DataFrame of shape (n_cpgs, n_samples).
    :param y: Unused; kept for a uniform filter interface.
    :param top_k: Number of CpGs to retain (default: 5000).
    :return: Boolean Series indexed by CpG, True for the top-k highest-variance features.
    """
    mask = _variance_top_k_mask(betas, top_k)
    logger.info("_variance_filter: %d / %d probes retained (top_k=%d)", mask.sum(), len(mask), top_k)
    return mask


def _differential_methylation_filter(
    betas: pd.DataFrame,
    y: pd.Series,
    top_k: int = 5000,
    task: Literal['classification', 'ordinal'] = 'classification',
) -> pd.Series:
    """
    Selects CpGs most associated with the target.

    For classification, uses an ANOVA F-test (``f_classif``) over discrete classes. For ordinal regression, ranks
    CpGs by absolute Spearman correlation with the continuous target — appropriate for an ordinal y where
    ``f_classif`` would treat each therapeutic group as an unordered category.

    :param betas: DataFrame of shape (n_cpgs, n_samples).
    :param y: Target aligned to the columns of ``betas``.
    :param top_k: Number of top CpGs to retain (default: 5000).
    :param task: ``'classification'`` (default) or ``'ordinal'``.
    :return: Boolean Series indexed by CpG, True for the top-k most associated probes.
    """
    X = betas.T
    if task == 'ordinal':
        # Per-CpG Spearman ρ via rank-transformed Pearson — O(n·m), avoids the m×m matrix that
        # `spearmanr` on a column-stacked 2D array would allocate (~3 TiB for 600k CpGs).
        y_rank = rankdata(np.asarray(y, dtype=float))
        x_rank = np.apply_along_axis(rankdata, 0, X.values)  # rank each CpG column independently
        y_c = y_rank - y_rank.mean()
        x_c = x_rank - x_rank.mean(axis=0)
        denom = np.sqrt((y_c ** 2).sum() * (x_c ** 2).sum(axis=0))
        rho = np.divide(x_c.T @ y_c, denom, out=np.zeros_like(denom), where=denom > 0)
        scores = pd.Series(np.abs(rho), index=betas.index)
        k = min(top_k, len(scores))
        cutoff = scores.nlargest(k).min()
        mask = scores >= cutoff
    else:
        sel = SelectKBest(f_classif, k=min(top_k, X.shape[1]))
        sel.fit(X, y)
        mask = pd.Series(sel.get_support(), index=betas.index)
    logger.info(
        "_differential_methylation_filter: %d / %d probes retained (task=%s, top_k=%d)",
        mask.sum(), len(mask), task, top_k,
    )
    return mask


def _boruta_filter(
    betas: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    pre_filter_top_k: int | None = 10_000,
    task: Literal['classification', 'ordinal'] = 'classification',
) -> pd.Series:
    """
    Selects CpGs using the Boruta all-relevant feature selection algorithm.

    Wraps a RandomForest (classifier or regressor depending on ``task``) in BorutaPy and returns a boolean
    mask of confirmed-relevant CpGs.
    Optionally applies a variance pre-filter before Boruta fitting.

    :param betas: DataFrame of shape (n_cpgs, n_samples).
    :param y: Target aligned to the columns of ``betas``.
    :param n_estimators: Number of trees in the underlying random forest (default: 100).
    :param pre_filter_top_k: If set, restrict to the top-k highest-variance CpGs before fitting Boruta (default: 10_000).
    :param task: ``'classification'`` (default) or ``'ordinal'``.
    :return: Boolean Series indexed by CpG, True for confirmed-relevant features.
    """
    full_index = betas.index

    if pre_filter_top_k is not None and pre_filter_top_k < len(betas):
        pre_mask = _variance_top_k_mask(betas, pre_filter_top_k)
        logger.info(
            "_boruta_filter: variance pre-filter retained %d / %d probes (top_k=%d)",
            pre_mask.sum(), len(pre_mask), pre_filter_top_k,
        )
        betas = betas.loc[pre_mask]

    X = betas.T.values  # BorutaPy requires numpy arrays
    if task == 'ordinal':
        # Regression Boruta rarely converges early on noisy ordinal targets and runs the full
        # max_iter at much higher per-iteration cost than the 3-class RF. Parallelise the RF and
        # halve the max iterations to keep runtime usable.
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
        selector = BorutaPy(rf, random_state=RANDOM_STATE, max_iter=50, verbose=1)
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
        selector = BorutaPy(rf, random_state=RANDOM_STATE)
    selector.fit(X, y.values)
    inner_mask = pd.Series(selector.support_, index=betas.index)

    # Expand back to the full feature space
    full_mask = pd.Series(False, index=full_index)
    full_mask.loc[inner_mask[inner_mask].index] = True

    logger.info(
        "_boruta_filter: %d / %d probes retained (task=%s, n_estimators=%s, pre_filter_top_k=%s)",
        full_mask.sum(), len(full_mask), task, n_estimators, pre_filter_top_k,
    )
    return full_mask


def _panel_promoters_filter(betas: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Selects CpGs that are part of the curated promoter panel.

    Loads the panel from config and matches against the betas index
    (expected format: "seqnames-start", e.g. "chr7-150800614").

    :param betas: DataFrame of shape (n_cpgs, n_samples), index = "seqnames-start".
    :param y: Class labels (unused; kept for a uniform filter interface).
    :return: Boolean Series indexed by betas.index, True for panel CpGs.
    """
    panel = pd.read_csv(_cfg_path('panel_promoters'), usecols=["genome_coordinates"])
    panel_set = set(panel["genome_coordinates"].dropna().unique())
    mask = pd.Series([cpg in panel_set for cpg in betas.index], index=betas.index)
    logger.info("_panel_promoters_filter: %d / %d probes retained", mask.sum(), len(mask))
    return mask


def _chrom_hmm_filter(betas: pd.DataFrame, y: pd.Series, states: list[str]) -> pd.Series:
    """
    Selects CpGs whose ChromHMM annotation matches one of the given states.

    Loads the ChromHMM annotation from config, builds a "seqnames-start" lookup key,
    and returns a boolean mask for probes whose state is in ``states``.

    :param betas: DataFrame of shape (n_cpgs, n_samples), index = "seqnames-start".
    :param y: Class labels (unused; kept for a uniform filter interface with run_cv).
    :param states: ChromHMM state labels to retain, e.g. ``["E13", "E14"]``.
    :return: Boolean Series indexed by betas.index, True for selected probes.
    """
    lookup = load_chrom_hmm_lookup()
    mask = pd.Series([lookup.get(cpg) in states for cpg in betas.index], index=betas.index)

    logger.info("_chrom_hmm_filter: %d / %d probes retained (states=%s)", mask.sum(), len(mask), states)

    return mask
