"""Canonical AJCC stage → therapeutic group mapping and metadata loading, shared across the ML pipeline.

Lives under :mod:`ml` because its consumers are here — :mod:`ml.tasks` (therapeutic group range),
:mod:`ml.features.cpg` (ordinal target), and the :mod:`ml.correlation` analysis module (ordinal grouping),
all of which import *from* this module, never the other way around.
"""
import pandas as pd

from ..config import _cfg_path


# Maps each raw AJCC stage label to its therapeutic group code.
AJCC_ORDER = {
    '0': 0, 'IA': 1, 'IB': 2, 'IIA': 3, 'IIB': 4, 'IIC': 4,
    'IIIA': 5, 'IIIB': 5, 'IIIC': 5,
}

TUMOR_DIAGNOSES = {'IM', 'NIM'}


def grouped_ajcc_labels() -> dict[int, str]:
    """Map each therapeutic group code to its combined AJCC-stage display label, so readers see the same groups the
    ordinal model predicts (``THERAPEUTIC_GROUPS``): ``4 -> 'IIB/C'``, ``5 -> 'III'``. Derived from
    :data:`AJCC_ORDER` so regrouping a stage there updates axis labels and legends everywhere at once.

    Sub-stages sharing a Roman numeral are shortened, since the fully spelled-out form does not fit the tick labels
    of a manuscript-width panel: to the numeral alone when the group holds *every* sub-stage of it
    (``IIIA/IIIB/IIIC -> III``), otherwise to the numeral plus the sub-stage letters (``IIB/IIC -> IIB/C``).
    """
    labels: dict[int, list[str]] = {}
    for label, code in AJCC_ORDER.items():
        labels.setdefault(code, []).append(label)

    grouped = {}
    for code, ls in labels.items():
        numeral = ls[0][:-1]  # the stage's Roman numeral, without its A/B/C sub-stage letter
        if len(ls) == 1 or not all(l.startswith(numeral) for l in ls):
            grouped[code] = '/'.join(ls)
        elif all(AJCC_ORDER[l] == code for l in AJCC_ORDER if l.startswith(numeral)):
            grouped[code] = numeral  # the whole stage is this one group, so its sub-stages need not be spelled out
        else:
            grouped[code] = ls[0] + ''.join('/' + l[len(numeral):] for l in ls[1:])
    return grouped


def load_ajcc_metadata() -> pd.DataFrame:
    """Load metadata filtered to IM + NIM with valid AJCC stadium, mapping each AJCC stage to its therapeutic group."""
    meta = pd.read_csv(_cfg_path('meta_data'))
    meta = meta[meta['groupedPrimaryDiagnosisPatho'].isin(TUMOR_DIAGNOSES)].copy()

    ajcc = meta['AJCC stadium'].astype(str).str.upper().str.strip()
    meta['ajcc_label'] = ajcc
    meta['therapeutic_group'] = meta['ajcc_label'].map(AJCC_ORDER)
    meta = meta.dropna(subset=['therapeutic_group']).copy()
    meta['therapeutic_group'] = meta['therapeutic_group'].astype(int)

    return meta.set_index('slideId')[['ajcc_label', 'therapeutic_group', 'groupedPrimaryDiagnosisPatho']].rename(
        columns={'groupedPrimaryDiagnosisPatho': 'diagnosis'}
    )
