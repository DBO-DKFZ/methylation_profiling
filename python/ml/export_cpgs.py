"""Export trained raw-CpG models' selected CpGs to ``selected_cpgs__*.csv``.

The R interpretation analyses (``R/analysis/dmr.R``, ``R/analysis/go_enrichment.R``) need the CpG list a model actually
selected, but the artifacts written by :mod:`ml.train` are joblib pickles of fitted estimators — unreadable from R
without a Python runtime. This module is the file-on-disk bridge, the same role :mod:`ml.export_folds` plays for the
per-fold CNV pipeline, and keeps the checkpoint the single source of truth instead of a hand-exported CSV.

Runs over every (task × filter × reducer × model) raw-CpG checkpoint by default; each ``--task`` / ``--filter`` /
``--reducer`` / ``--model`` flag narrows the sweep. Output filenames mirror the artifact they came from, so the exported
selection is always traceable to one checkpoint.

Only raw-CpG artifacts qualify: for the markers/stacked feature sources ``selected_cpgs`` holds feature names, not CpGs.
Note that with a reducer in the pipeline the selection is still the post-filter, pre-reduction CpG list — the CpGs the
reducer consumed.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from ..config import MODELS_DIR
from ._registries import FILTERS, REDUCERS
from .tasks import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def find_artifacts(
    task: Optional[str] = None,
    filter_name: Optional[str] = None,
    reducer_name: Optional[str] = None,
    model_name: Optional[str] = None,
    models_dir: Path = MODELS_DIR,
) -> list[Path]:
    """Locate trained raw-CpG artifacts, optionally narrowed by pipeline component.

    Each argument left as None becomes a wildcard in the ``model__<task>__cpg__<filter>__<reducer>__<model>.pkl``
    filename that :mod:`ml.train` writes. The ``cpg`` segment is fixed, so markers/stacked artifacts never match.

    :param task: Task name to restrict to, or None for all.
    :param filter_name: CpG filter name to restrict to, or None for all.
    :param reducer_name: Reducer name to restrict to, or None for all.
    :param model_name: Model name to restrict to, or None for all.
    :param models_dir: Directory holding the ``model__*.pkl`` artifacts.
    :return: Matching artifact paths, sorted by filename.
    :raises FileNotFoundError: If nothing matches.
    """
    pattern = (f'model__{task or "*"}__cpg__{filter_name or "*"}'
               f'__{reducer_name or "*"}__{model_name or "*"}.pkl')
    matches = sorted(models_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No artifact matching {pattern} in {models_dir}. Train one first, e.g. '
            f'`python -m python.ml.train --task classification --features cpg --filter boruta --model svm`.'
        )
    return matches


def default_out_path(artifact_path: Path) -> Path:
    """Sibling CSV path mirroring the artifact's (task, features, filter, reducer, model) stem."""
    return artifact_path.with_name(artifact_path.stem.replace('model__', 'selected_cpgs__', 1) + '.csv')


def export_cpgs(artifact_path: Path, out_path: Optional[Path] = None) -> pd.Series:
    """Read a trained artifact's selected CpGs and optionally write them as a one-column CSV.

    :param artifact_path: Path to a ``model__*.pkl`` written by :mod:`ml.train`.
    :param out_path: If given, write the CpGs there as a single ``genome_coordinates`` column.
    :return: The selected CpGs, in the order the model sees them.
    :raises ValueError: If the artifact was not trained on the raw-CpG feature source.
    """
    artifact = joblib.load(artifact_path)

    features = artifact.get('features')
    if features != 'cpg':
        raise ValueError(
            f"{artifact_path.name} was trained on the '{features}' feature source; its 'selected_cpgs' holds feature "
            f'names rather than CpGs. Only raw-CpG artifacts can be exported.'
        )

    cpgs = pd.Series(list(artifact['selected_cpgs']), name='genome_coordinates')
    logger.info('%s (%s / %s / %s): %d selected CpGs.', artifact_path.name, artifact.get('filter_name'),
                artifact.get('reducer_name'), artifact.get('model_name'), len(cpgs))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cpgs.to_csv(out_path, index=False)
        logger.info('Selected CpGs saved to %s.', out_path)

    return cpgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default=None,
                        help='Restrict to one task (default: all).')
    parser.add_argument('--filter', default=None,
                        help=f'Restrict to one CpG filter (default: all). One of: {list(FILTERS)} or "none".')
    parser.add_argument('--reducer', default=None,
                        help=f'Restrict to one reducer (default: all). One of: {list(REDUCERS)} or "none".')
    parser.add_argument('--model', default=None,
                        help='Restrict to one model name (default: all).')
    parser.add_argument('--artifact', type=Path, default=None,
                        help='Explicit artifact path, bypassing discovery and the narrowing flags.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV path (default: the artifact path with model__ -> selected_cpgs__ and a .csv '
                             'suffix). Requires --artifact.')
    args = parser.parse_args()

    if args.out is not None and args.artifact is None:
        parser.error('--out applies to a single artifact; pass --artifact as well.')

    artifacts = [args.artifact] if args.artifact else find_artifacts(
        args.task, args.filter, args.reducer, args.model,
    )
    logger.info('Exporting %d artifact(s).', len(artifacts))
    for artifact_path in artifacts:
        export_cpgs(artifact_path, out_path=args.out or default_out_path(artifact_path))


if __name__ == '__main__':
    main()
