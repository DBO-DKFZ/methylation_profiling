"""Task definitions for the unified CV pipeline.

A :class:`Task` bundles everything that differs between classification (IM/NIM/NV 3-class) and ordinal regression
(therapeutic groups, sourced from :data:`ml.ajcc.AJCC_ORDER`): target column in the metadata, how to stratify CV
splits, which metrics to compute, which sklearn predict method to call, and the OOF schema.

The CV loop in :mod:`ml.cv` reads everything task-specific from this object so its body has no
``task_type == 'ordinal_regression'`` branches.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from .ajcc import AJCC_ORDER


# Therapeutic group range — derived from the canonical mapping so dropping a stage in one place updates QWK clipping,
# per-group MAE metrics, and confusion-matrix labels everywhere at once. Deduplicated because grouped stages share a
# code (IIB/IIC -> 4, IIIA/IIIB/IIIC -> 5) and would otherwise repeat as label rows/columns.
THERAPEUTIC_GROUPS = sorted(set(AJCC_ORDER.values()))


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
# Each takes (y_true, y_pred, y_prob) — y_prob is None for ordinal regression.
# Returning a Python float keeps DataFrame columns dtype-stable.

def _auroc(y_true, y_pred, y_prob):
    return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))


def _bal_acc(y_true, y_pred, y_prob):
    return float(balanced_accuracy_score(y_true, y_pred))


def _f1_macro(y_true, y_pred, y_prob):
    return float(f1_score(y_true, y_pred, average='macro'))


def _auroc_per_class(y_true, y_pred, y_prob):
    return roc_auc_score(y_true, y_prob, multi_class='ovr', average=None)


def _f1_per_class(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred, labels=list(range(y_prob.shape[1])), average=None)


def _per_group_mae(y_true, y_pred) -> pd.Series:
    return pd.Series(np.abs(y_true.to_numpy() - y_pred), index=y_true.index).groupby(y_true).mean()


def _mae_macro(y_true, y_pred, y_prob):
    return float(_per_group_mae(y_true, y_pred).mean())


def _qwk(y_true, y_pred, y_prob):
    # QWK on integer labels; clip continuous regressor output to the therapeutic group range.
    y_pred_int = np.clip(np.round(y_pred), THERAPEUTIC_GROUPS[0], THERAPEUTIC_GROUPS[-1]).astype(int)
    return float(cohen_kappa_score(y_true.astype(int), y_pred_int, labels=THERAPEUTIC_GROUPS, weights='quadratic'))


def _spearman_rho(y_true, y_pred, y_prob):
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def _make_mae_group(group: int) -> Callable:
    def _mae(y_true, y_pred, y_prob):
        per_group = _per_group_mae(y_true, y_pred)
        v = per_group.get(group, np.nan)
        return float(v) if pd.notna(v) else np.nan
    _mae.__name__ = f'mae_group_{group}'
    return _mae


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Bundle of task-specific behaviour driving :func:`ml.cv.run_cv`.

    :param name: ``'classification'`` or ``'ordinal'``.
    :param target_col: Metadata column holding the target.
    :param metric_fns: Mapping metric name -> ``fn(y_true, y_pred, y_prob) -> float``. ``y_prob`` is ``None`` when the
        model has no ``predict_proba``.
    :param per_class_metric_fns: Mapping base name -> ``fn(y_true, y_pred, y_prob) -> array`` returning one value per
        class (aligned to ``class_names``). Expanded into ``<base>_<class>`` metrics with bootstrap CIs by
        :func:`ml.evaluate.bootstrap_ci_metrics`.
    :param headline_metrics: Subset of metric names highlighted in :func:`ml.cv.summarize`.
    :param oof_schema: ``'probs'`` makes ``prob_<class>`` columns the primary output and the metric prediction the
        argmax; ``'prediction'`` scores on the model's own ``predict`` output (a single ``prediction`` column).
        ``'prediction'``-schema models that also expose ``predict_proba`` additionally get ``prob_<class>`` columns
        written for downstream stacking — the metrics still use ``predict``.
    :param predict_method: ``'predict_proba'`` (classification) or ``'predict'`` (ordinal).
    """
    name: Literal['classification', 'ordinal']
    target_col: str
    metric_fns: dict[str, Callable] = field(default_factory=dict)
    per_class_metric_fns: dict[str, Callable] = field(default_factory=dict)
    headline_metrics: tuple[str, ...] = ()
    oof_schema: Literal['probs', 'prediction'] = 'probs'
    predict_method: Literal['predict_proba', 'predict'] = 'predict_proba'
    class_names: tuple[str, ...] | None = None


CLASSIFICATION = Task(
    name='classification',
    target_col='groupedPrimaryDiagnosisPatho',
    metric_fns={
        'auroc':    _auroc,
        'bal_acc':  _bal_acc,
        'f1_macro': _f1_macro,
    },
    per_class_metric_fns={
        'auroc': _auroc_per_class,
        'f1':    _f1_per_class,
    },
    headline_metrics=('auroc', 'bal_acc', 'f1_macro'),
    oof_schema='probs',
    predict_method='predict_proba',
    class_names=('IM', 'NIM', 'NV'),
)


ORDINAL = Task(
    name='ordinal',
    target_col='therapeutic_group',
    metric_fns={
        'mae_macro':    _mae_macro,
        'qwk':          _qwk,
        'spearman_rho': _spearman_rho,
        **{f'mae_group_{g}': _make_mae_group(g) for g in THERAPEUTIC_GROUPS},
    },
    headline_metrics=('mae_macro', 'qwk', 'spearman_rho'),
    oof_schema='prediction',
    predict_method='predict',
)


TASKS: dict[str, Task] = {
    'classification': CLASSIFICATION,
    'ordinal':        ORDINAL,
}
