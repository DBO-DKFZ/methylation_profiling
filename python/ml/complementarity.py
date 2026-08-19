"""Sample-level complementarity of the two stacking base learners, on top of unified OOF artefacts.

Diffs two OOF prediction files (a CpG one and a markers one, both from :mod:`ml.oof`); reads each file's own
``prediction``/``y_true`` so it is task-agnostic. Two metrics select what "one learner does better" means:

* ``exact`` (default, for the 3-class diagnosis): a 2×2 (CpG correct?) × (markers correct?) contingency table on
  exact-match correctness.
* ``ordinal`` (for the 6-way therapeutic group target): compares per-sample distance to the truth ``|pred - y_true|``
  and counts the cases where one learner lands strictly *closer* than the other. This respects the ordinal structure
  the exact match throws away — landing one group off is not the same as five off.

The analysis is purely descriptive: whether the two learners *differ* is already settled upstream by the paired
model comparison on the primary endpoint (:mod:`ml.compare` — AUROC for classification, MAE for the ordinal task).
What this module adds is *where* they differ, i.e. which samples one learner rescues for the other.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import COMPLEMENTARITY_DIR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def complementarity(cpg_oof_path: Path, marker_oof_path: Path, metric: str = 'exact') -> pd.DataFrame:
    """Sample-level comparison of the two base learners.

    Diffs two OOF prediction files (both from :mod:`ml.oof` under the same fold structure) rather than refitting
    anything: the markers side is the markers classifier's own OOF CSV (``--features markers``). Each file's own
    ``prediction``/``y_true`` columns are used, so the comparison is task-agnostic.

    :param cpg_oof_path: Path to a CpG-classifier OOF CSV.
    :param marker_oof_path: Path to a markers-classifier OOF CSV (same task/split as ``cpg_oof_path``).
    :param metric: ``exact`` scores each prediction as an exact class match (nominal diagnosis); ``ordinal`` compares
        distance to the truth and counts which learner lands strictly closer (therapeutic groups).
    :return: DataFrame indexed by ``slideId`` with per-sample CpG/markers predictions and, per metric, correctness
        flags (``exact``) or absolute errors and closer-than flags (``ordinal``).
    """
    cpg = _read_predictions(cpg_oof_path, 'cpg')
    mrk = _read_predictions(marker_oof_path, 'markers')
    if 'y_true' not in cpg.columns:
        raise ValueError(f'{cpg_oof_path} has no y_true column to score against.')

    common = cpg.index.intersection(mrk.index)
    if len(common) == 0:
        raise ValueError('CpG and marker OOF files share no slideIds.')
    cpg, mrk = cpg.loc[common], mrk.loc[common]

    df = pd.DataFrame(index=common)
    df.index.name = 'slideId'
    df['y_true'] = cpg['y_true']
    if 'y_true' in mrk.columns and (mrk['y_true'].to_numpy() != df['y_true'].to_numpy()).any():
        logger.warning('CpG and marker OOF y_true disagree — are these the same task/target?')
    df['cpg_pred'] = cpg['cpg_pred']
    df['markers_pred'] = mrk['markers_pred']
    for src, prefix in ((cpg, 'cpg_prob_'), (mrk, 'markers_prob_')):
        for c in src.columns:
            if c.startswith(prefix):
                df[c] = src[c]
    df['fold'] = mrk['fold'] if 'fold' in mrk.columns else cpg.get('fold')
    logger.info('Compared %d samples from CpG vs marker predictions.', len(df))

    if metric == 'ordinal':
        df['cpg_abs_err'] = (df['cpg_pred'] - df['y_true']).abs()
        df['markers_abs_err'] = (df['markers_pred'] - df['y_true']).abs()
        df['cpg_closer'] = df['cpg_abs_err'] < df['markers_abs_err']
        df['markers_closer'] = df['markers_abs_err'] < df['cpg_abs_err']
        _log_closer_summary(df)
    else:
        df['cpg_correct'] = df['cpg_pred'] == df['y_true']
        df['markers_correct'] = df['markers_pred'] == df['y_true']
        _log_complementarity_summary(df)
    return df


def _read_predictions(path: Path, label: str) -> pd.DataFrame:
    """Read an OOF prediction CSV into a frame indexed by ``slideId`` with a ``{label}_pred`` column (plus ``y_true``
    and ``fold`` when present, and any ``prob_*`` columns renamed ``{label}_prob_*``).

    Prefers the file's own ``prediction`` column when present — for ordinal models the decision is a cumulative-
    threshold rule, not the argmax of the marginal ``prob_*`` — and falls back to the argmax of ``prob_*`` (with the
    integer suffix as the class label) for classification files that carry only probabilities.
    """
    df = pd.read_csv(path, index_col='slideId')
    df.index = df.index.astype(str)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if 'prediction' in df.columns:
        pred = df['prediction'].astype(int).to_numpy()
    elif prob_cols:
        labels = np.array([int(c.split('_')[1]) for c in prob_cols])
        pred = labels[df[prob_cols].to_numpy().argmax(axis=1)]
    else:
        raise ValueError(f'{path} has neither a "prediction" nor "prob_*" column.')

    out = pd.DataFrame({f'{label}_pred': pred.astype(int)}, index=df.index)
    for c in prob_cols:
        out[f'{label}_{c}'] = df[c]
    for c in ('y_true', 'fold'):
        if c in df.columns:
            out[c] = df[c].astype(int)
    return out


def _log_complementarity_summary(df: pd.DataFrame) -> None:
    both_right = (df['cpg_correct'] & df['markers_correct']).sum()
    both_wrong = (~df['cpg_correct'] & ~df['markers_correct']).sum()
    cpg_only = (df['cpg_correct'] & ~df['markers_correct']).sum()
    markers_only_right = (~df['cpg_correct'] & df['markers_correct']).sum()

    logger.info('=== AGREEMENT TABLE (n=%d) ===', len(df))
    logger.info('  both correct           : %4d (%.1f%%)', both_right, 100 * both_right / len(df))
    logger.info('  CpG correct, markers no: %4d (%.1f%%)', cpg_only, 100 * cpg_only / len(df))
    logger.info('  CpG wrong, markers yes : %4d (%.1f%%)  ← complementary rescues',
                markers_only_right, 100 * markers_only_right / len(df))
    logger.info('  both wrong             : %4d (%.1f%%)', both_wrong, 100 * both_wrong / len(df))

    discordant = cpg_only + markers_only_right
    logger.info('  discordant pairs        : %4d (of which %d are markers rescues)', discordant, markers_only_right)

    cpg_wrong = df[~df['cpg_correct']]
    logger.info('=== RESCUE RATE on CpG mistakes by true class ===')
    for cls in sorted(df['y_true'].unique()):
        sub = cpg_wrong[cpg_wrong['y_true'] == cls]
        if len(sub) == 0:
            continue
        rescued = sub['markers_correct'].sum()
        logger.info('  class %d: %3d / %3d CpG mistakes correctly classified by markers (%.1f%%)',
                    cls, rescued, len(sub), 100 * rescued / len(sub))

    logger.info('=== Inverse: CpG rescue rate on markers mistakes by true class ===')
    markers_wrong = df[~df['markers_correct']]
    for cls in sorted(df['y_true'].unique()):
        sub = markers_wrong[markers_wrong['y_true'] == cls]
        if len(sub) == 0:
            continue
        rescued = sub['cpg_correct'].sum()
        logger.info('  class %d: %3d / %3d markers mistakes correctly classified by CpG (%.1f%%)',
                    cls, rescued, len(sub), 100 * rescued / len(sub))


def _log_closer_summary(df: pd.DataFrame) -> None:
    """Ordinal analogue of :func:`_log_complementarity_summary`: a four-way split of every sample into both correct,
    both wrong by an equal distance, CpG strictly closer, or markers strictly closer."""
    n = len(df)
    cpg_closer = int(df['cpg_closer'].sum())
    markers_closer = int(df['markers_closer'].sum())
    both_correct = int(((df['cpg_abs_err'] == 0) & (df['markers_abs_err'] == 0)).sum())
    both_wrong = n - cpg_closer - markers_closer - both_correct  # equal distance, both off the true group

    logger.info('=== CLOSER-TO-TRUTH TABLE (n=%d) ===', n)
    logger.info('  both correct           : %4d (%.1f%%)', both_correct, 100 * both_correct / n)
    logger.info('  both wrong, equal dist : %4d (%.1f%%)', both_wrong, 100 * both_wrong / n)
    logger.info('  CpG closer             : %4d (%.1f%%)', cpg_closer, 100 * cpg_closer / n)
    logger.info('  markers closer         : %4d (%.1f%%)', markers_closer, 100 * markers_closer / n)

    decisive = cpg_closer + markers_closer
    logger.info('  decisive pairs         : %4d (of which %d are markers closer)', decisive, markers_closer)

    logger.info('=== who lands closer, by true group ===')
    for cls in sorted(df['y_true'].unique()):
        sub = df[df['y_true'] == cls]
        c, m = int(sub['cpg_closer'].sum()), int(sub['markers_closer'].sum())
        logger.info('  group %d: CpG closer %3d | markers closer %3d | tie %3d  (n=%d)',
                    cls, c, m, len(sub) - c - m, len(sub))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpg-pred', type=Path, required=True,
                        help='Path to a CpG-classifier prediction CSV (OOF from ml.oof or external-test predictions).')
    parser.add_argument('--markers-pred', type=Path, required=True,
                        help='Path to a markers-classifier prediction CSV (same task/split as --cpg-pred).')
    parser.add_argument('--metric', choices=('exact', 'ordinal'), default='exact',
                        help='exact class match (nominal diagnosis) or ordinal closer-to-truth counts (therapeutic groups).')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    df = complementarity(args.cpg_pred, args.markers_pred, metric=args.metric)
    out = args.out or COMPLEMENTARITY_DIR / f'complementarity__{args.metric}__{args.cpg_pred.stem}__vs__{args.markers_pred.stem}.csv'

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)
    logger.info('Wrote %s', out)


if __name__ == '__main__':
    main()
