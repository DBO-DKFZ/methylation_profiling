"""FeatureSource protocol: anything that yields (X, y) for a given task and may swap in fold-specific references in
:meth:`prepare_fold`.

Three implementations live alongside in this package:

* :class:`ml.features.cpg.CpGFeatures` — raw CpG betas (~644k columns); supports the filter/reducer pipeline.
* :class:`ml.features.markers.MarkerFeatures` — 12 CpG-derived markers (Horvath EAA, EpiScore cell types, CNV burden);
  per-fold CNV references are swapped in :meth:`prepare_fold`.
* :class:`ml.features.stacked.StackedFeatures` — CpG-classifier prediction columns joined with the marker
  classifier's prediction columns (two-base-learner stacking).
"""
from typing import Protocol

import pandas as pd

from ..tasks import Task


class FeatureSource(Protocol):
    """Protocol for things that produce a (X, y) feature matrix for a :class:`Task`.

    :ivar supports_cpg_pipeline: True only for raw CpG inputs — the CV loop will skip the
        filter/reducer step otherwise (markers/stacked features are already low-dim and
        meaningfully named).
    """

    supports_cpg_pipeline: bool

    def load(self, task: Task) -> tuple[pd.DataFrame, pd.Series]:
        """Load the full (X, y) for ``task``. Subsequent fold-specific overrides are applied
        by :meth:`prepare_fold`.
        """
        ...

    def prepare_fold(
        self,
        fold: int,
        train_idx: pd.Index,
        val_idx: pd.Index,
        X: pd.DataFrame,
        y: pd.Series,
        task: Task,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return per-fold (X, y) — default is identity. CNV-aware sources override this to
        swap the validation rows' CNV burden to the fold's train-only reference.
        """
        return X, y
