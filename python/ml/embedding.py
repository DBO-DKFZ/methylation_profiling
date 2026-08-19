"""Unsupervised two-dimensional embedding of the CpG samples, coloured by each task's target, for both tasks.

Embedded are the 500 CpGs the cohort-overview heatmap draws, persisted by ``R/analysis/cohort_heatmap.R`` as the top of
a per-CpG Kruskal-Wallis ranking against the diagnosis. This is the scatter counterpart of that heatmap: same samples,
same CpGs, so the two figures are directly comparable.

Unlike :mod:`ml.correlation`, this runs on the **whole cohort** rather than the training split. The point of the figure
is to describe the cohort. Train and test samples are distinguished by marker shape.

Both :mod:`ml.tasks` targets get their own embedding.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.manifold import TSNE

from ..config import COHORT_DIR, COHORT_PLOTS, DIAGNOSIS_COLORS, RANDOM_STATE, THERAPEUTIC_COLORS
from ..visualization import plot_embedding
from .ajcc import grouped_ajcc_labels
from .correlation import _load_group_metadata
from .features.markers import _split_slide_ids
from .tasks import TASKS, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


PERPLEXITY = 30

# Both tasks' panels sit side by side in one row of the assembled cohort figure, so they are drawn at one fixed aspect
# instead of each following its own cloud (see :func:`visualization.plot_embedding`).
PANEL_ASPECT = 0.32

# The slice R/analysis/cohort_heatmap.R writes: rows = genome coordinates, one column per slideId.
COHORT_SLICE = COHORT_DIR / 'classification_cpgs_betas.csv'


def _load_cohort_slice(task: Task) -> pd.DataFrame:
    """The cohort-overview heatmap's CpGs, restricted to the samples ``task`` covers.

    Reads the slice R writes. The task's sample set comes from the same grouping metadata the colours do, which is the
    filtering :class:`~ml.features.cpg.CpGFeatures` applies to the full matrices.

    :param task: Task definition, selecting which samples are in scope.
    :return: Betas of shape (n_samples, n_cpgs), indexed by ``slideId``.
    """
    if not COHORT_SLICE.exists():
        raise FileNotFoundError(f'No cohort beta slice at {COHORT_SLICE}. Run R/analysis/cohort_heatmap.R first.')
    X = pd.read_csv(COHORT_SLICE, index_col=0).T
    X.index = X.index.astype(str)
    in_scope = X.index.intersection(_load_group_metadata(task)[0].index)
    logger.info('[%s] cohort slice: %d of %d samples in scope, %d CpGs.',
                task.name, len(in_scope), X.shape[0], X.shape[1])
    return X.loc[in_scope]


def embed(X: pd.DataFrame, perplexity: float = PERPLEXITY) -> pd.DataFrame:
    """Embed the samples in two dimensions with t-SNE.

    :param X: Betas of shape (n_samples, n_cpgs), indexed by ``slideId``.
    :param perplexity: Requested t-SNE perplexity, clamped to ``(n_samples - 1) / 3``.
    :return: Frame indexed like ``X`` with ``dim1``/``dim2`` columns.
    """
    features = X.to_numpy()

    effective = min(perplexity, (X.shape[0] - 1) / 3)
    if effective < perplexity:
        logger.info('Perplexity lowered from %g to %g for %d samples.', perplexity, effective, X.shape[0])
    tsne = TSNE(n_components=2, perplexity=effective, init='pca', random_state=RANDOM_STATE)
    coords = tsne.fit_transform(features)
    logger.info('t-SNE on (%d, %d) converged after %d iterations, KL divergence = %.3f',
                *features.shape, tsne.n_iter_, tsne.kl_divergence_)
    return pd.DataFrame(coords, index=X.index, columns=['dim1', 'dim2'])


def run_all(task: Task, results_dir: Path, plots_dir: Path, perplexity: float = PERPLEXITY) -> None:
    """Embed the cohort for ``task``, persist the coordinates, and draw the scatter.

    The coordinates are written before plotting so restyling the panel never re-runs the embedding. Both file names
    carry the task.

    :param task: Task definition, selecting the sample set and the grouping variable points are coloured by.
    :param results_dir: Directory for ``tsne__<task>.csv``; created if missing.
    :param plots_dir: Directory for ``tsne__<task>.pdf``; created if missing.
    :param perplexity: Requested t-SNE perplexity.
    :return: None
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    X = _load_cohort_slice(task)
    split = pd.Series('train', index=X.index, name='split')
    split[X.index.intersection(_split_slide_ids('test'))] = 'test'
    logger.info('[%s] %d samples (%s).', task.name, X.shape[0],
                ', '.join(f'{k}={v}' for k, v in split.value_counts().items()))

    meta, order, is_ordinal = _load_group_metadata(task)
    coords = embed(X, perplexity)
    coords = coords.join(meta['group']).join(split)
    coords.index.name = 'slideId'

    csv_path = results_dir / f'tsne__{task.name}.csv'
    coords.to_csv(csv_path)
    logger.info('[%s] wrote %s', task.name, csv_path)

    # Ordinal colors are keyed by group code, the labels by their combined AJCC stage, so bridge the two.
    if is_ordinal:
        colors = {label: THERAPEUTIC_COLORS[code] for code, label in grouped_ajcc_labels().items()}
    else:
        colors = DIAGNOSIS_COLORS

    plot_path = plots_dir / f'tsne__{task.name}.pdf'
    plot_embedding(coords, order, colors, aspect=PANEL_ASPECT, out_path=plot_path)
    logger.info('[%s] wrote %s', task.name, plot_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default=None,
                        help='Task to embed (default: run both).')
    parser.add_argument('--perplexity', type=float, default=PERPLEXITY,
                        help='t-SNE perplexity; lowered automatically when the cohort is too small.')
    parser.add_argument('--results-dir', type=Path, default=COHORT_DIR)
    parser.add_argument('--plots-dir', type=Path, default=COHORT_PLOTS)
    args = parser.parse_args()

    for name in ([args.task] if args.task else list(TASKS)):
        run_all(TASKS[name], args.results_dir, args.plots_dir, args.perplexity)


if __name__ == '__main__':
    main()
