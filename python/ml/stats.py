"""Statistical tests for the analysis pipelines.

The project's home for hypothesis tests: each function delegates the actual computation to an established library
(:mod:`scipy.stats`, :mod:`statsmodels`, :mod:`MLstatkit`) and adds only project-uniform return shapes, effect-size
formulas, the one-vs-rest / pairwise loops, and edge-case handling. Two families live here:

* **Model comparison** (:mod:`ml.compare`) — AUROC and MAE tests between two trained models on the external test split.
* **Marker–target group comparison** (:mod:`ml.correlation`) — Kruskal-Wallis with an ε² effect size and, conditional
  on it, post-hoc pairwise Mann-Whitney U.

Model comparison provides two flavours, matching the two tasks:

* **AUROC (classification)** — split by how many classes each metric spans. The **macro-OvR** AUROC is compared with a
  *paired* difference bootstrap via :func:`scipy.stats.bootstrap` (DeLong is inherently binary and has no maintained
  multiclass implementation for the macro average), clustered on patient (:func:`_group_rows`) so a patient's slides
  are resampled together. Each **per-class one-vs-rest** AUROC is binary, so it uses **DeLong's test**
  (:func:`MLstatkit.Delong_test`) — the standard analytic comparison of two correlated binary ROC curves; being
  analytic it has no patient-clustered form here and treats slides as independent. Both yield a CI on the *difference*
  plus a p-value; unlike the per-model marginal CIs in :func:`ml.evaluate.bootstrap_ci_metrics`, overlapping marginal
  CIs do not settle whether two models differ, a paired test does.
* **MAE (ordinal)** — the **macro-averaged** MAE is the reported endpoint, and like macro AUROC it is an average over
  groups (here the true therapeutic groups) that no analytic paired test addresses, so it uses the same
  patient-clustered paired difference bootstrap (:func:`paired_bootstrap_mae_diff`).

Multiple comparisons are corrected with Holm-Bonferroni (:func:`holm_correction`) throughout — the model-comparison
families, the correlation post-hoc pairwise family, and (from the caller) the correlation across-marker omnibus family.
"""
import logging
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from MLstatkit import Delong_test
from scipy.stats import DegenerateDataWarning, bootstrap, kruskal, mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from ..config import RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# Overdraw factor for :func:`paired_bootstrap_mae_diff`: it draws this multiple of the requested resamples so that,
# after discarding those that lost a whole true-label group (a few percent), the requested number still remains.
_OVERDRAW = 2


# ---------------------------------------------------------------------------
# Shared machinery — paired difference bootstrap over patient clusters
# ---------------------------------------------------------------------------

def _safe(fn, *args) -> float:
    """Call ``fn(*args)``, returning NaN when a metric is undefined for the (resampled) sample — e.g. a class missing
    from a bootstrap draw makes ``roc_auc_score`` raise. Mirrors the skip-and-continue guard in
    :func:`ml.evaluate.bootstrap_ci_metrics`."""
    try:
        return float(fn(*args))
    except ValueError:
        return float('nan')


def _summarize_diff(deltas: np.ndarray) -> tuple[float, float, float]:
    """Percentile 95% CI and a two-sided percentile-bootstrap p-value from a distribution of metric differences.

    The p-value is ``2 * min(P(Δ* ≤ 0), P(Δ* ≥ 0))`` (clipped to 1) — the proportion of resamples on the null side of
    zero, doubled — consistent with the CI-inversion view (significant at 0.05 ⟺ the 95% CI excludes 0).
    """
    if len(deltas) < 2:
        return float('nan'), float('nan'), float('nan')
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    p_value = min(1.0, 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return ci_low, ci_high, p_value


def _group_rows(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by cluster label, in first-appearance order — the unit resampled by the patient-clustered
    (block) bootstrap.

    Resampling whole clusters (patients) rather than individual rows keeps a patient's correlated slides together, so
    the bootstrap does not treat them as independent observations (which understates variance). Order is preserved
    (``sort=False``) so that when every row is its own cluster the resampling reduces *exactly* to the ordinary
    row-level bootstrap — clustering is then a true no-op, not a reshuffle.

    :param groups: 1-D per-row cluster labels (e.g. patient IDs), aligned to the data rows.
    :return: List of row-index arrays (one per distinct label), in first-appearance order.
    """
    rows = pd.Series(np.arange(len(groups)))
    return [g.to_numpy() for _, g in rows.groupby(np.asarray(groups), sort=False)]


def _paired_bootstrap_diff(
    metric,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    key: str,
    groups: np.ndarray | None,
    n_resamples: int,
    random_state: int,
) -> dict[str, dict[str, float]]:
    """Paired difference bootstrap of any macro metric between two models on the same samples.

    Shared by :func:`paired_bootstrap_auroc_diff` and :func:`paired_bootstrap_mae_diff`. One resample is drawn per
    iteration and applied to *both* models, so the difference distribution captures their correlation. When ``groups``
    is given the resampling is a patient-clustered (block) bootstrap — whole patients are drawn with replacement
    (:func:`_group_rows`) so same-patient slides are not treated as independent; otherwise it is the ordinary per-row
    bootstrap.

    ``_OVERDRAW * n_resamples`` resamples are drawn and the first ``n_resamples`` for which ``metric`` was defined are
    kept, so the CI and p-value rest on the full requested count even when some resamples are unusable (a class or
    label group missing from the draw). This is an overdraw, not a bias correction: the retained resamples are still
    conditioned on the metric being computable, exactly as discarding them would be.

    :param metric: ``metric(y_true, pred) -> float`` returning NaN when undefined for the (resampled) sample, which
        marks that resample unusable.
    :param y_true: 1-D ground-truth labels.
    :param pred_a: Predictions for model A (1-D or ``(n_samples, n_classes)``), aligned to ``y_true``.
    :param pred_b: Predictions for model B, aligned to ``pred_a``.
    :param key: Metric name to key the returned row by (e.g. ``'auroc_macro'``).
    :param groups: Optional 1-D cluster labels (patient IDs) aligned to ``y_true``; clusters the resampling.
    :param n_resamples: Usable bootstrap resamples to summarise.
    :param random_state: Seed for the bootstrap RNG.
    :return: ``{key: {test, value_a, value_b, delta, ci_low, ci_high, p_value, n}}``. ``delta = value_a - value_b``;
        ``n`` is the sample count, while the resampling unit is patients when clustered.
    """
    n = len(y_true)
    rows = _group_rows(np.arange(n) if groups is None else np.asarray(groups))
    codes = np.arange(len(rows))

    value_a = metric(y_true, pred_a)
    value_b = metric(y_true, pred_b)

    def _diff(csample: np.ndarray) -> float:
        i = np.concatenate([rows[c] for c in csample.astype(int)])
        return metric(y_true[i], pred_a[i]) - metric(y_true[i], pred_b[i])

    # The DegenerateDataWarning scipy raises whenever the distribution contains NaNs is suppressed: it reports that
    # *its* CI is unavailable, and the CI here comes from `_summarize_diff` over the usable deltas instead.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DegenerateDataWarning)
        res = bootstrap((codes,), _diff, n_resamples=_OVERDRAW * n_resamples, method='percentile',
                        vectorized=False, rng=np.random.default_rng(random_state))
    deltas = res.bootstrap_distribution
    usable = deltas[np.isfinite(deltas)][:n_resamples]
    if len(usable) < len(deltas):
        logger.info('%s bootstrap: %d/%d drawn resamples discarded as the metric was undefined for them.',
                    key, len(deltas) - int(np.isfinite(deltas).sum()), len(deltas))
    if len(usable) < n_resamples:
        logger.warning('%s bootstrap: only %d/%d usable resamples — CI and p-value rest on fewer resamples than '
                       'requested; raise _OVERDRAW.', key, len(usable), n_resamples)
    ci_low, ci_high, p_value = _summarize_diff(usable)
    return {key: {
        'test': 'paired cluster bootstrap' if groups is not None else 'paired bootstrap',
        'value_a': value_a, 'value_b': value_b, 'delta': value_a - value_b,
        'ci_low': ci_low, 'ci_high': ci_high, 'p_value': p_value, 'n': n,
    }}


# ---------------------------------------------------------------------------
# AUROC — macro-OvR paired bootstrap + DeLong (per-class binary OvR)
# ---------------------------------------------------------------------------

def _macro_ovr_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC — the same definition as ``ml.tasks._auroc`` so values match test_metrics.json.
    NaN when a resample leaves a class unrepresented, which makes ``roc_auc_score`` raise."""
    return _safe(lambda y, p: roc_auc_score(y, p, multi_class='ovr', average='macro'), y_true, y_prob)


def paired_bootstrap_auroc_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Paired difference bootstrap of macro-OvR AUROC between two models on the same samples.

    Thin metric-specific front end to :func:`_paired_bootstrap_diff`, which documents the resampling scheme. Both
    probability matrices must be aligned to ``y_true`` (same rows, same order).

    :param y_true: 1-D integer ground-truth labels (class index).
    :param prob_a: ``(n_samples, n_classes)`` probabilities for model A.
    :param prob_b: ``(n_samples, n_classes)`` probabilities for model B, aligned to ``prob_a``.
    :param groups: Optional 1-D cluster labels (patient IDs) aligned to ``y_true``; clusters the resampling.
    :param n_resamples: Bootstrap resamples (default 1000).
    :param random_state: Seed for the bootstrap RNG.
    :return: ``{'auroc_macro': {test, value_a, value_b, delta, ci_low, ci_high, p_value, n}}``. ``delta = value_a -
        value_b`` (positive ⇒ A higher).
    """
    return _paired_bootstrap_diff(
        _macro_ovr_auroc, np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b), 'auroc_macro',
        groups=groups, n_resamples=n_resamples, random_state=random_state,
    )


def delong_auroc_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    class_names: tuple[str, ...],
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Per-class one-vs-rest DeLong test comparing two models' correlated binary AUCs (via :func:`MLstatkit.Delong_test`).

    Each class is binarised one-vs-rest and DeLong compares the two models' AUCs on that binary problem. The 95% CI on
    the difference is ``delta ± 1.96·sqrt(var_diff)`` using the AUC-difference variance DeLong reports; MLstatkit falls
    back to an internal bootstrap on degeneracy (invalid variance), in which case the CI is left NaN.

    :param y_true: 1-D integer ground-truth labels (class index).
    :param prob_a: ``(n_samples, n_classes)`` probabilities for model A.
    :param prob_b: ``(n_samples, n_classes)`` probabilities for model B, aligned to ``prob_a``.
    :param class_names: Class labels in column order (drives the ``auroc_<class>`` keys).
    :param random_state: Seed for MLstatkit's bootstrap fallback.
    :return: ``{auroc_<class>: {value_a, value_b, delta, ci_low, ci_high, statistic, p_value, n}}``. ``statistic`` is
        the signed z aligned to ``delta = value_a - value_b`` (positive ⇒ A higher).
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    results: dict[str, dict[str, float]] = {}
    for cls, name in enumerate(class_names):
        _z, p_value, _ci_a, _ci_b, auc_a, auc_b, info = Delong_test(
            (y_true == cls).astype(int), prob_a[:, cls], prob_b[:, cls], random_state=random_state,
        )
        delta = float(auc_a) - float(auc_b)
        var = info.get('var_diff')
        se = float(np.sqrt(var)) if var is not None and var > 0 else float('nan')
        if info.get('method') == 'bootstrap':
            logger.info('DeLong degenerate for class %r — MLstatkit used its bootstrap fallback.', name)
        results[f'auroc_{name}'] = {
            'test': 'DeLong', 'value_a': float(auc_a), 'value_b': float(auc_b), 'delta': delta,
            'ci_low': delta - 1.96 * se, 'ci_high': delta + 1.96 * se,
            'statistic': delta / se if np.isfinite(se) and se > 0 else float('nan'),
            'p_value': float(p_value), 'n': n,
        }
    return results


# ---------------------------------------------------------------------------
# MAE — paired difference bootstrap (macro-averaged over the true-label groups)
# ---------------------------------------------------------------------------

def paired_bootstrap_mae_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Paired difference bootstrap of macro-averaged MAE between two models on the same samples.

    Metric-specific front end to :func:`_paired_bootstrap_diff`, which documents the resampling scheme. A bootstrap is
    used because the macro average is taken over the true-label groups, which no analytic paired test addresses.

    A resample that loses a whole true-label group is treated as unusable (NaN) rather than averaged over the surviving
    groups, which would silently redefine the macro average mid-bootstrap.

    :param y_true: 1-D ground-truth ordinal labels.
    :param pred_a: 1-D predictions for model A, aligned to ``y_true``.
    :param pred_b: 1-D predictions for model B, aligned to ``pred_a``.
    :param metric_fn: Macro-MAE callable with the ``ml.tasks`` metric signature ``(y_true, y_pred, y_prob)``, taking
        ``y_true`` as a :class:`pandas.Series` — pass ``task.metric_fns['mae_macro']` so the definition stays single-sourced.
    :param groups: Optional 1-D cluster labels (patient IDs) aligned to ``y_true``; clusters the resampling.
    :param n_resamples: Bootstrap resamples (default 1000).
    :param random_state: Seed for the bootstrap RNG.
    :return: ``{'mae_macro': {test, value_a, value_b, delta, ci_low, ci_high, p_value, n}}``. ``delta = value_a -
        value_b`` (negative ⇒ A lower, i.e. better, MAE).
    """
    y_true = np.asarray(y_true)
    n_labels = len(np.unique(y_true))

    def _macro_mae(y: np.ndarray, pred: np.ndarray) -> float:
        if len(np.unique(y)) < n_labels:   # a whole group dropped out — macro average not comparable
            return float('nan')
        return _safe(lambda yy, pp: metric_fn(pd.Series(yy), pp, None), y, pred)

    return _paired_bootstrap_diff(
        _macro_mae, y_true, np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float), 'mae_macro',
        groups=groups, n_resamples=n_resamples, random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Group comparison — Kruskal-Wallis (+ ε²) and post-hoc pairwise Mann-Whitney U
# ---------------------------------------------------------------------------

def kruskal_effect_size(*samples: np.ndarray) -> dict[str, float]:
    """Kruskal-Wallis H test across independent groups, with the epsilon-squared effect size.

    ε² = ``(H - k + 1) / (n - k)`` rescales the H statistic to ``[0, 1]`` (0 = no rank separation between groups,
    1 = complete separation), a sample-size-independent companion to the p-value. Returns NaNs (rather than raising)
    when fewer than two groups are supplied or ``n <= k``, mirroring the degenerate-case guards elsewhere in this
    module.

    :param samples: Two or more 1-D per-group value arrays (already filtered to the groups worth testing).
    :return: ``{statistic, p_value, effect_size}`` with ``effect_size`` the Kruskal-Wallis ε².
    """
    groups = [np.asarray(s, dtype=float) for s in samples]
    if len(groups) < 2:
        return {'statistic': float('nan'), 'p_value': float('nan'), 'effect_size': float('nan')}
    h, p = kruskal(*groups)
    k = len(groups)
    n = sum(g.size for g in groups)
    eps2 = (float(h) - k + 1) / (n - k) if n > k else float('nan')
    return {'statistic': float(h), 'p_value': float(p), 'effect_size': eps2}


def pairwise_mannwhitney(groups: dict[str, np.ndarray], order: list[str]) -> list[dict]:
    """Post-hoc pairwise Mann-Whitney U across groups, Holm-adjusted over the pair family.

    Every unordered pair of groups present in ``groups`` is compared with a two-sided Mann-Whitney U test; the raw
    p-values form one family corrected with :func:`holm_correction`. Pairs follow ``order`` so the comparison order is
    stable, and labels in ``order`` but absent from ``groups`` are skipped.

    The reported effect size is the common-language effect size ``U / (n1 * n2)`` — the probability that a random
    sample from ``group1`` exceeds one from ``group2`` (0.5 = no rank separation). It is the raw U rescaled to
    ``[0, 1]``, which the raw statistic needs to be comparable across pairs of unequal size.

    :param groups: Mapping of group label to its 1-D value array (only groups worth testing, e.g. ``n >= 2``).
    :param order: Group labels in display order, fixing the pair enumeration.
    :return: One dict per pair — ``{group1, group2, statistic, p_value, p_adj, effect_size, n}`` — in ``order``'s pair
        sequence.
    """
    pairs = list(combinations([g for g in order if g in groups], 2))
    tests = [mannwhitneyu(groups[g1], groups[g2], alternative='two-sided') for g1, g2 in pairs]
    p_adj = holm_correction([t.pvalue for t in tests])
    return [
        {'group1': g1, 'group2': g2, 'statistic': float(t.statistic), 'p_value': float(t.pvalue),
         'p_adj': float(padj), 'effect_size': float(t.statistic) / (groups[g1].size * groups[g2].size),
         'n': int(groups[g1].size + groups[g2].size)}
        for (g1, g2), t, padj in zip(pairs, tests, p_adj)
    ]


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_correction(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control) via :func:`statsmodels...multipletests`.

    NaN entries (undefined comparisons) are excluded from the family and returned as NaN, so they neither inflate the
    correction nor get an adjusted value.

    :param pvalues: Sequence of raw p-values forming one comparison family.
    :return: Array of adjusted p-values, same length/order as the input.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    mask = np.isfinite(pvalues)
    if mask.any():
        adjusted[mask] = multipletests(pvalues[mask], method='holm')[1]
    return adjusted
