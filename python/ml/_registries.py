"""Shared filter and reducer registries used by :mod:`ml.benchmark` and :mod:`ml.train`.

Keeping these in one place ensures both the CV benchmark and the single-model training entry
point see the same set of named filters / reducers — add a filter once and both commands can use it.
"""
import logging
from functools import partial

from scipy.stats import iqr

from .filters import (
    _boruta_filter, _chrom_hmm_filter, _differential_methylation_filter,
    _elasticnet_filter, _panel_promoters_filter, _variance_filter,
)
from .models import MODELS
from .reducers import (
    AutoEncoderReducer, ChromHMMReducer, PCAReducer, PLSReducer,
    VariationalAutoEncoderReducer,
)

logger = logging.getLogger(__name__)


FILTERS = {
    'iqr':                      lambda betas, y=None, threshold=0.2: betas.apply(iqr, axis=1) > threshold,
    'lasso':                    partial(_elasticnet_filter, l1_ratio=1.0),
    'boruta':                   _boruta_filter,
    'elasticnet':               _elasticnet_filter,
    'variance':                 _variance_filter,
    'differential_methylation': _differential_methylation_filter,
    'panel_promoters':          _panel_promoters_filter,
}
FILTERS.update({
    f'chrom_hmm_{x}{y}': partial(_chrom_hmm_filter, states=[x, y])
    for x, y in [('E13', 'E14'), ('E6', 'E7'), ('E1', 'E2')]
})


REDUCERS = {
    'none':         (None,                          None),
    'chrom_hmm':    (ChromHMMReducer,               {'agg': 'mean'}),
    'pca100':       (PCAReducer,                    {'n_components': 100}),
    'pca200':       (PCAReducer,                    {'n_components': 200}),
    'pca300':       (PCAReducer,                    {'n_components': 300}),
    'pls20':        (PLSReducer,                    {'n_components': 20}),
    'pls30':        (PLSReducer,                    {'n_components': 30}),
    'ae':           (AutoEncoderReducer,            {'latent_dim': 64, 'hidden_dims': (512, 256),
                                                     'epochs': 10, 'batch_size': 32}),
    'ae_vae':       (VariationalAutoEncoderReducer, {'latent_dim': 64, 'hidden_dims': (512, 256),
                                                     'epochs': 10, 'batch_size': 32,
                                                     'beta': 1.0}),
}


def resolve_pipeline(
    task_name: str,
    features: str,
    model_name: str,
    filter_name: str,
    reducer_name: str,
    supports_cpg_pipeline: bool,
) -> tuple:
    """Resolve CLI registry names into concrete pipeline components. Shared by the single-model
    entry points :mod:`ml.train` and :mod:`ml.oof` so both validate and gate identically.

    Validates ``model_name`` against the task/feature-specific block of :data:`ml.models.MODELS`,
    and — for raw-CpG feature sources — ``filter_name``/``reducer_name`` against :data:`FILTERS`/
    :data:`REDUCERS`. Low-dimensional sources (markers, stacked) are already named and skip both
    steps, so filter/reducer are forced to ``'none'``.

    :param task_name: Task key into :data:`ml.models.MODELS`.
    :param features: Feature-source name ('cpg', 'markers', 'stacked'); selects the model block.
    :param model_name: Model registry key for the resolved (task, feature) block.
    :param filter_name: CpG filter name or 'none'.
    :param reducer_name: Reducer name (registry key), 'none' included.
    :param supports_cpg_pipeline: Whether the feature source runs the CpG filter/reducer steps.
    :return: ``(model, filter_fn, filter_name, reducer, reducer_name)`` where ``reducer`` is a
        ``(cls, kwargs)`` tuple (``(None, None)`` when disabled) and ``filter_fn`` is ``None`` when disabled.
    """
    feature_kind = 'cpg' if features == 'cpg' else 'lowdim'
    if model_name not in MODELS[task_name][feature_kind]:
        raise ValueError(
            f'Unknown model {model_name!r} for task={task_name}, features={features}. '
            f'Available: {list(MODELS[task_name][feature_kind])}'
        )
    model = MODELS[task_name][feature_kind][model_name]

    if supports_cpg_pipeline:
        if filter_name != 'none' and filter_name not in FILTERS:
            raise ValueError(f'Unknown filter {filter_name!r}. Available: {list(FILTERS)} or "none".')
        if reducer_name not in REDUCERS:
            raise ValueError(f'Unknown reducer {reducer_name!r}. Available: {list(REDUCERS)}.')
        filter_fn = FILTERS[filter_name] if filter_name != 'none' else None
        reducer = REDUCERS[reducer_name]
    else:
        if filter_name != 'none' or reducer_name != 'none':
            logger.info('--features %s ignores filter/reducer; forcing both to "none".', features)
        filter_name, reducer_name = 'none', 'none'
        filter_fn = None
        reducer = (None, None)

    return model, filter_fn, filter_name, reducer, reducer_name


__all__ = ['FILTERS', 'REDUCERS', 'resolve_pipeline']
