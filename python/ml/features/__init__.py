from pathlib import Path

from .base import FeatureSource
from .cpg import CpGFeatures
from .markers import MarkerFeatures
from .stacked import StackedFeatures


FEATURE_SOURCES: dict[str, type[FeatureSource]] = {
    'cpg':     CpGFeatures,
    'markers': MarkerFeatures,
    'stacked': StackedFeatures,
}

# Display names for the feature sources, used wherever a figure names the model it shows (plot titles in
# :mod:`ml.evaluate` / :mod:`ml.importance`, row labels in :mod:`ml.compare`). The feature source is what differs
# between the compared models, so the learner is left out of the name and stays in the figure caption.
SOURCE_LABELS: dict[str, str] = {
    'cpg':     'CpG-based',
    'markers': 'Marker-based',
    'stacked': 'Stacked',
}


def build(features: str, cpg_oof_path: Path | None = None, split: str = 'train',
          marker_oof_path: Path | None = None) -> FeatureSource:
    """Construct a :class:`FeatureSource` from the ``--features`` CLI flag.

    :param features: One of ``'cpg'``, ``'markers'``, ``'stacked'``.
    :param cpg_oof_path: Required when ``features == 'stacked'`` — path to a CpG-classifier OOF CSV.
    :param split: ``'train'`` or ``'test'``. Honoured by all sources: CpG selects the matching betas file, markers/
        stacked partition samples by the config test clinic (mirroring the CpG split) so external-test evaluation scores
        unseen samples.
    :param marker_oof_path: Required when ``features == 'stacked'`` — path to a marker-classifier prediction CSV used
        as the second view (two-base-learner stacking).
    :return: Instantiated feature source.
    :raises ValueError: If ``features`` is unknown or ``cpg_oof_path`` / ``marker_oof_path`` is missing for stacked.
    """
    if features == 'cpg':
        return CpGFeatures(split=split)
    if features == 'markers':
        return MarkerFeatures(split=split)
    if features == 'stacked':
        if cpg_oof_path is None or marker_oof_path is None:
            raise ValueError('--features stacked requires --cpg-oof <path> and --marker-oof <path>')
        return StackedFeatures(cpg_oof_path, marker_oof_path=marker_oof_path)
    raise ValueError(f'Unknown features: {features}')


__all__ = ['FeatureSource', 'CpGFeatures', 'MarkerFeatures', 'StackedFeatures', 'FEATURE_SOURCES', 'SOURCE_LABELS',
           'build']
