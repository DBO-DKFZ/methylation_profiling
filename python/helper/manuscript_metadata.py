"""Summarize study-cohort metadata for the manuscript.

Clinic names in meta.csv are anonymized to hospitalN identifiers using data/hospital_mapping.json before anything is
written or logged.

Outputs (under results/manuscript/):
  - patient_overview.csv      patient-level summary (demographics: age, sex, skin type)
  - lesion_overview.csv       lesion-level summary (localization, diagnosis, AJCC, Breslow)
  - diagnosis_by_hospital.csv diagnosis counts per anonymized hospital
  - therapeutic_group_by_hospital.csv
                              therapeutic group counts per anonymized hospital (IM/NIM with
                              a valid AJCC stadium, i.e. the ordinal task's cohort)
  - clinical_table.csv        Table 1: demographics & localization stratified by
                              train / test split (config.yaml 'test_clinic') and diagnosis
"""
import json
import logging

import pandas as pd

from ..config import _CONFIG, _cfg_path, MANUSCRIPT_DIR
from ..ml.ajcc import AJCC_ORDER, TUMOR_DIAGNOSES, grouped_ajcc_labels

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
HOSPITAL_MAPPING: dict[str, str] = json.loads(_cfg_path('hospital_mapping').read_text())

# The external-test hospitals follow the pipeline's own train/test split (config.yaml 'test_clinic',
# see preprocessing._train_test_split) so this table can never disagree with the models' split.
_TEST_CLINICS = _CONFIG['test_clinic']
TEST_HOSPITALS: list[str] = [
    HOSPITAL_MAPPING[c] for c in ([_TEST_CLINICS] if isinstance(_TEST_CLINICS, str) else _TEST_CLINICS)
]

DIAGNOSIS_ORDER = ['IM', 'NIM', 'NV']
MELANOMA_DIAGNOSES = sorted(TUMOR_DIAGNOSES)  # canonical IM + NIM set, shared with the ordinal task

SEX_LABELS = {0: 'male', 1: 'female'}
SEX_ORDER = ['male', 'female']

AGE_BIN_ORDER = ['<35', '35-54', '55-74', '>74', 'Unknown']
SKINTYPE_ORDER = ['I', 'II', 'III', 'IV', 'V', 'VI', 'Unknown']

LOCATION_MAP = {
    'face': 'Face/scalp/neck', 'scalp': 'Face/scalp/neck', 'neck': 'Face/scalp/neck',
    'palms': 'Palms/soles', 'soles': 'Palms/soles',
    'arm': 'Upper extremities', 'forearm': 'Upper extremities', 'hand': 'Upper extremities',
    'thigh': 'Lower extremities', 'leg (knee and below)': 'Lower extremities', 'foot': 'Lower extremities',
    'back': 'Back', 'abdomen': 'Abdomen', 'chest': 'Chest',
    'buttock': 'Buttock', 'genitalia': 'Genitalia',
}
LOCATION_ORDER = [
    'Face/scalp/neck', 'Palms/soles', 'Upper extremities', 'Lower extremities',
    'Back', 'Abdomen', 'Chest', 'Buttock', 'Genitalia', 'Unknown',
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _age_bin(age: float) -> str:
    if pd.isna(age):
        return 'Unknown'
    if age < 35:
        return '<35'
    if age <= 54:
        return '35-54'
    if age <= 74:
        return '55-74'
    return '>74'


def _fmt_cell(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else '0 (-)'


def _numeric_stats(series: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(series, errors='coerce')
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return {
        f'{prefix}_mean': s.mean(),
        f'{prefix}_median': s.median(),
        f'{prefix}_std': s.std(),
        f'{prefix}_q1': q1,
        f'{prefix}_q3': q3,
        f'{prefix}_iqr': q3 - q1,
        f'{prefix}_min': s.min(),
        f'{prefix}_max': s.max(),
        f'{prefix}_n_missing': int(s.isna().sum()),
    }


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['age_bin'] = df['approxAge'].apply(_age_bin)
    df['sex_label'] = df['sex'].map(SEX_LABELS)
    df['skin_label'] = df['skinType'].astype(str).str.upper()
    df.loc[~df['skin_label'].isin(SKINTYPE_ORDER), 'skin_label'] = 'Unknown'
    df['loc_label'] = (
        df['location'].astype(str).str.lower().map(LOCATION_MAP).fillna('Unknown')
    )
    return df


def _hospital_range(hospitals: list[str]) -> str:
    """'hospital 8' / 'hospitals 1-7' / 'hospitals 1, 3, 5' for a set of hospitalN identifiers."""
    nums = sorted(int(h.removeprefix('hospital')) for h in hospitals)
    if len(nums) == 1:
        return f'hospital {nums[0]}'
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f'hospitals {nums[0]}-{nums[-1]}'
    return 'hospitals ' + ', '.join(str(n) for n in nums)


def _count_rows(prefix: str, col: str, order: list[str], df: pd.DataFrame, denom: int) -> list[tuple]:
    counts = df[col].value_counts()
    rows = []
    for label in order:
        n = int(counts.get(label, 0))
        pct = round(100 * n / denom, 2) if denom else ''
        rows.append((f'{prefix}_{label}', n, pct))
    return rows


# ── Data loading ─────────────────────────────────────────────────────────────
def load_cohort() -> pd.DataFrame:
    """Load meta.csv restricted to samples with methylation data and anonymize clinic."""
    meta = pd.read_csv(_cfg_path('meta_data'))
    samples = pd.read_csv(_cfg_path('methylation_samples_cleaned'))

    cohort = meta.merge(samples[['slideId']].drop_duplicates(), on='slideId', how='inner')
    cohort['hospital'] = cohort['clinic'].astype(str).str.lower().map(HOSPITAL_MAPPING)

    missing = cohort.loc[cohort['hospital'].isna(), 'clinic'].unique()
    if len(missing):
        raise ValueError(f"Clinics missing from hospital_mapping.json: {sorted(missing)}")

    cohort = cohort.drop(columns=['clinic']).rename(columns={'gender': 'sex'})
    return cohort[cohort['groupedPrimaryDiagnosisPatho'] != 'other'].copy()


# ── Table builders ───────────────────────────────────────────────────────────
def build_patient_overview(cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per patient — demographics (age, sex, skin type)."""
    df = _add_labels(cohort).drop_duplicates(subset='patientId')
    n = len(df)
    rows: list[tuple] = [('n_patients', n, '')]
    rows += [(k, v, '') for k, v in _numeric_stats(df['approxAge'], 'age').items()]
    rows += _count_rows('age_bin', 'age_bin', AGE_BIN_ORDER, df, n)
    rows += _count_rows('sex', 'sex_label', SEX_ORDER, df, n)
    rows += _count_rows('skinType', 'skin_label', SKINTYPE_ORDER, df, n)
    return pd.DataFrame(rows, columns=['metric', 'value', 'percent'])


def build_lesion_overview(cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion — localization, diagnosis, AJCC, Breslow."""
    df = _add_labels(cohort).drop_duplicates(subset='lesionId')
    n = len(df)
    rows: list[tuple] = [
        ('n_lesions', n, ''),
        ('n_hospitals', df['hospital'].nunique(), ''),
    ]
    rows += _count_rows('location', 'loc_label', LOCATION_ORDER, df, n)
    rows += _count_rows('diagnosis', 'groupedPrimaryDiagnosisPatho', DIAGNOSIS_ORDER, df, n)

    melanoma = df[df['groupedPrimaryDiagnosisPatho'].isin(MELANOMA_DIAGNOSES)]
    n_m = len(melanoma)
    rows.append(('n_melanoma', n_m, ''))
    for k, v in melanoma['AJCC stadium'].value_counts(dropna=False).items():
        label = 'missing' if pd.isna(k) else str(k)
        rows.append((f'ajcc_{label}', int(v), round(100 * v / n_m, 2) if n_m else ''))
    rows += [(k, v, '') for k, v in _numeric_stats(melanoma['Breslow thickness'], 'breslow').items()]
    return pd.DataFrame(rows, columns=['metric', 'value', 'percent'])


def _counts_by_hospital(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    """Counts of ``col`` per anonymized hospital with column-wise percentages and margins."""
    counts = pd.crosstab(df['hospital'], df[col], margins=True, margins_name='Total')
    rows = sorted(ix for ix in counts.index if ix != 'Total') + ['Total']
    counts = counts.reindex(index=rows, columns=order + ['Total'], fill_value=0)
    denominators = counts.loc['Total']  # column totals; for 'Total' col = grand total

    out = pd.DataFrame(index=counts.index)
    for c in counts.columns:
        out[c] = [_fmt_cell(int(n), int(denominators[c])) for n in counts[c]]
    return out


def build_diagnosis_by_hospital(cohort: pd.DataFrame) -> pd.DataFrame:
    """Diagnosis counts per anonymized hospital with column-wise percentages."""
    return _counts_by_hospital(cohort, 'groupedPrimaryDiagnosisPatho', DIAGNOSIS_ORDER)


def build_therapeutic_group_by_hospital(cohort: pd.DataFrame) -> pd.DataFrame:
    """Therapeutic group counts per anonymized hospital, over the same cohort the ordinal task trains on:
    IM + NIM tumours with a valid AJCC stadium, grouped by :data:`ml.ajcc.AJCC_ORDER`."""
    labels = grouped_ajcc_labels()
    df = cohort[cohort['groupedPrimaryDiagnosisPatho'].isin(MELANOMA_DIAGNOSES)].copy()
    group = df['AJCC stadium'].astype(str).str.upper().str.strip().map(AJCC_ORDER)
    df = df.assign(group_label=group.map(labels)).dropna(subset=['group_label'])
    return _counts_by_hospital(df, 'group_label', [labels[code] for code in sorted(labels)])


def build_clinical_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """Manuscript Table 1: demographics stratified by split × diagnosis."""
    df = _add_labels(cohort)
    df['split'] = df['hospital'].apply(lambda h: 'Test' if h in TEST_HOSPITALS else 'Training')

    splits = ['Training', 'Test']
    cols = ['Overall'] + DIAGNOSIS_ORDER
    split_headers = {
        sp: f'{sp} set ({_hospital_range(sorted(df.loc[df["split"] == sp, "hospital"].unique()))})'
        for sp in splits
    }

    subsets: dict[tuple[str, str], pd.DataFrame] = {}
    for sp in splits:
        s = df[df['split'] == sp]
        subsets[(sp, 'Overall')] = s
        for d in DIAGNOSIS_ORDER:
            subsets[(sp, d)] = s[s['groupedPrimaryDiagnosisPatho'] == d]
    col_keys = [(sp, d) for sp in splits for d in cols]
    totals = {k: len(v) for k, v in subsets.items()}

    # (category_header, subcategory_label, source_column, source_value)
    # source_column=None marks a section-header row with empty data cells.
    sections: list[tuple[str, str, str | None, str | None]] = [
        ('Age at diagnosis (years)', '', None, None),
        *[('', b, 'age_bin', b) for b in AGE_BIN_ORDER[:-1]],  # drop 'Unknown'
        ('Sex', '', None, None),
        *[('', g, 'sex_label', g) for g in SEX_ORDER],
        ('Fitzpatrick skin type', '', None, None),
        *[('', st, 'skin_label', st) for st in SKINTYPE_ORDER
          if st != 'VI' or (df['skin_label'] == 'VI').any()],
        ('Lesion localization', '', None, None),
        *[('', loc, 'loc_label', loc) for loc in LOCATION_ORDER],
    ]

    header_labels = {
        (sp, d): (split_headers[sp] if d == 'Overall' else '',
                  f'{d}\nn={totals[(sp, d)]}')
        for sp in splits for d in cols
    }
    columns = pd.MultiIndex.from_tuples(
        [('', 'Category'), ('', 'Subcategory')] + [header_labels[k] for k in col_keys]
    )

    out_rows = []
    for category, label, src, val in sections:
        if src is None:
            out_rows.append([category, label] + [''] * len(col_keys))
        else:
            cells = [_fmt_cell(int((subsets[k][src] == val).sum()), totals[k]) for k in col_keys]
            out_rows.append([category, label] + cells)
    return pd.DataFrame(out_rows, columns=columns)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    logger.info(f"Cohort size (methylation samples): {len(cohort)}")
    logger.info(f"Samples per hospital:\n{cohort['hospital'].value_counts().sort_index().to_string()}")
    logger.info(
        f"Overall diagnosis counts:\n"
        f"{cohort['groupedPrimaryDiagnosisPatho'].value_counts(dropna=False).to_string()}"
    )

    tables = {
        'patient_overview.csv':      (build_patient_overview(cohort),      False),
        'lesion_overview.csv':       (build_lesion_overview(cohort),       False),
        'diagnosis_by_hospital.csv': (build_diagnosis_by_hospital(cohort), True),
        'therapeutic_group_by_hospital.csv': (build_therapeutic_group_by_hospital(cohort), True),
        'clinical_table.csv':        (build_clinical_table(cohort),        False),
    }
    for name, (table, write_index) in tables.items():
        path = MANUSCRIPT_DIR / name
        table.to_csv(path, index=write_index)
        logger.info(f"Wrote {path}")
