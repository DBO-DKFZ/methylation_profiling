from pathlib import Path
from typing import Optional

import yaml


def _load_config(path: Optional[Path] = None) -> dict:
    """Load YAML configuration.

    Search order:
    - Explicit path if provided
    - python/config.yaml (next to this file)
    - project root config.yaml (parent directory of this file)
    """
    if path is None:
        # 1) python/config.yaml
        path = Path(__file__).with_name('config.yaml')
        if not path.exists():
            # 2) project root config.yaml
            candidate = Path(__file__).resolve().parents[1] / 'config.yaml'
            if candidate.exists():
                path = candidate

    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


_CONFIG = _load_config()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

RANDOM_STATE: int = _CONFIG['random_state']
DIAGNOSIS_COLORS: dict[str, str] = _CONFIG['colors']['diagnosis']

# Therapeutic group colors keyed by AJCC-derived group code. Group 0 reuses the NIM
# diagnosis color (it is essentially NIM); 1-5 come from the config's blue ramp.
THERAPEUTIC_COLORS: dict[int, str] = {
    0: DIAGNOSIS_COLORS['NIM'],
    **{int(code): color for code, color in _CONFIG['colors']['therapeutic'].items()},
}


# Figure geometry/typography from the target journal's guidelines, shared with R/lib/plot_utils.R. Applied to
# matplotlib's rcParams by visualization.py, which also turns `panel_mm` slots into figure sizes.
FIGURE: dict = _CONFIG['figure']


def _cfg_path(key: str) -> Path:
    """Resolve a path from the YAML config's 'paths' section relative to project root."""
    rel = _CONFIG['paths'][key]
    return (PROJECT_ROOT / rel).resolve()


# Output roots come from config; per-analysis subdirs are derived here so the directory
# layout is defined in one place rather than spread across config keys and call sites.
RESULTS_DIR: Path = _cfg_path('results')
PLOTS_DIR: Path = _cfg_path('plots')

# Cross-package analysis dirs (written by R, read by ml/correlation.py, ml/features/markers.py, ml/embedding.py).
CNV_BURDEN_DIR: Path = RESULTS_DIR / 'cnv_burden'

# Cohort description: the heatmap R draws and the sample embedding ml/embedding.py
COHORT_DIR: Path = RESULTS_DIR / 'cohort'
COHORT_PLOTS: Path = PLOTS_DIR / 'cohort'

# Cohort description tables for the manuscript (helper/manuscript_metadata.py).
MANUSCRIPT_DIR: Path = RESULTS_DIR / 'manuscript'

# Classifier outputs, subdivided by artifact kind.
CLASSIFIER_DIR: Path = RESULTS_DIR / 'classifier'
MODELS_DIR: Path = CLASSIFIER_DIR / 'models'             # trained model__*.pkl
CV_DIR: Path = CLASSIFIER_DIR / 'cv'                      # cv_results*, checkpoints, cv_folds, oof_predictions
PREDICTIONS_DIR: Path = CLASSIFIER_DIR / 'predictions'   # external-test test_predictions__*
IMPORTANCE_DIR: Path = CLASSIFIER_DIR / 'importance'     # permutation_*, chromhmm_selection_enrichment
COMPLEMENTARITY_DIR: Path = CLASSIFIER_DIR / 'complementarity'  # complementarity__* (CpG↔markers base-learner diffs)
COMPARISON_DIR: Path = CLASSIFIER_DIR / 'comparison'     # comparison__* (pairwise model significance tables)

CLASSIFIER_PLOTS: Path = PLOTS_DIR / 'classifier'        # eval plots stay at CLASSIFIER_PLOTS/<artifact_stem>
IMPORTANCE_PLOTS: Path = CLASSIFIER_PLOTS / 'importance'  # permutation_*.pdf
COMPARISON_PLOTS: Path = CLASSIFIER_PLOTS / 'comparison'  # comparison__*.pdf (Δmetric forest plots)

# Marker <-> target correlation analysis (ml/correlation.py); per-task subdirs correlation/<task>/.
CORRELATION_DIR: Path = RESULTS_DIR / 'correlation'
CORRELATION_PLOTS: Path = PLOTS_DIR / 'correlation'

# Assembled multi-panel manuscript figures (panels.py), composited from the plots the analyses wrote elsewhere here.
MANUSCRIPT_PLOTS: Path = PLOTS_DIR / 'manuscript'
