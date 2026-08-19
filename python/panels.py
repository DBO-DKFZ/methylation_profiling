"""
Assembles the multi-panel manuscript figures from the individual plot PDFs the analyses write.

Every plot is already drawn at the width of the slot it occupies here (see :func:`visualization.panel_size`), so this
script only *places* pages — each panel goes down at scale 1.0, keeping text from the R and Python figures at the same
size. Nothing is stretched to fit; an oversized figure warns rather than rescaling, and the fix belongs upstream.

Panel letters are stamped here as bold lower-case a/b/c, so the individual plots stay reusable in any position.

Usage:
    python -m python.panels                  # rebuild every figure
    python -m python.panels --figure deconvolution   # just one
"""

import argparse
import logging
from pathlib import Path

import pymupdf
from matplotlib import font_manager

from .config import FIGURE, MANUSCRIPT_PLOTS, PLOTS_DIR

logger = logging.getLogger(__name__)

MM = 72 / 25.4        # PDF points per mm
GUTTER_MM = 2.0       # between panels, horizontally and between rows
LABEL_STRIP_MM = 3.0  # blank band above each row holding that row's panel letters (an 8 pt cap is ~2.8 mm)

# One entry per manuscript figure: rows of plot paths relative to PLOTS_DIR, laid out left-to-right and top-to-bottom.
# Panel letters follow that same order. Rows are independent — as tall as their tallest entry, centred horizontally —
# so re-grouping a figure is just moving paths between the lists below. A row entry may itself be a list, stacking
# those panels in one column of the row, which is how a tall panel is paired with two short ones.
_CLS_MODELS = {  # classification: per-model plot directory, in the order their panels should appear
    'cpg': 'classifier/model__classification__cpg__boruta__none__svm',
    'markers': 'classifier/model__classification__markers__none__none__random_forest',
    'stacked': 'classifier/model__classification__stacked__none__none__mlp',
}
_ORD_MODELS = {
    'cpg': 'classifier/model__ordinal__cpg__boruta__none__ogboost',
    'markers': 'classifier/model__ordinal__markers__none__none__ogboost',
    'stacked': 'classifier/model__ordinal__stacked__none__none__mord_ridge',
}
_IMPORTANCE = 'classifier/importance'

FIGURES: dict[str, list[list[str | list[str]]]] = {
    # 1. Discrimination per model (one row, all three models side by side), then what the selected CpGs are enriched
    # for beside their beta values. The heatmap is also a panel of the DMR figure below - the same page in both.
    'classification_auroc_go': [
        [f'{d}/roc_curves.pdf' for d in _CLS_MODELS.values()],
        ['go/classification/go_top_terms.pdf', 'dmr/heatmap_selected_high_effect.pdf'],
    ],
    # 2. Where each classification model's errors land.
    'classification_confusion': [
        [f'{d}/confusion_matrix.pdf' for d in _CLS_MODELS.values()],
    ],
    # 3. Ordinal error by group and which features drive it. Importance is shown per feature and by feature block for
    # the markers model, but only by block for the stacked one, whose inputs are the base learners' class
    # probabilities — the blocks (CpG view / marker view) are the interpretable unit there.
    'ordinal_performance': [
        [f'{d}/per_group_mae.pdf' for d in _ORD_MODELS.values()],
        [f'{_IMPORTANCE}/permutation__model__ordinal__markers__none__none__ogboost.pdf',
         [f'{_IMPORTANCE}/permutation_grouped__model__ordinal__markers__none__none__ogboost.pdf',
          f'{_IMPORTANCE}/permutation_grouped__model__ordinal__stacked__none__none__mord_ridge.pdf']],
    ],
    # 4. Where each ordinal model's errors land.
    'ordinal_confusion': [
        [f'{d}/confusion_matrix.pdf' for d in _ORD_MODELS.values()],
    ],
    # 5. Differential methylation across the diagnoses: the called regions' effect sizes and how many high-effect
    # regions each contrast yields, then where NIM falls on the NV -> IM axis in each direction. Both rows are computed
    # over the whole cohort (train+test).
    'differential_methylation': [
        ['dmr/dmr_volcano.pdf', 'dmr/dmr_counts_per_contrast.pdf'],
        ['nim_spectrum/axis_gain.pdf', 'nim_spectrum/axis_loss.pdf'],
    ],
    # 6/7. Marker-vs-target association, one panel per modality — epigenetic age acceleration, CNV burden, and the
    # EpiSCORE cell types as a figure-wide 3x3 row — closed by that task's cross-marker summary heatmap (pairwise
    # Mann-Whitney U effect size for classification, signed Spearman ρ for ordinal). All three boxplot panels share one
    # facet aspect (ml.correlation.FACET_ASPECT), set so this fits one page.
    'classification_markers': [
        ['correlation/classification/horvath_eaa.pdf', 'correlation/classification/cnv.pdf'],
        ['correlation/classification/episcore.pdf'],
        ['correlation/classification/heatmap.pdf'],
    ],
    'ordinal_markers': [
        ['correlation/ordinal/horvath_eaa.pdf', 'correlation/ordinal/cnv.pdf'],
        ['correlation/ordinal/episcore.pdf'],
        ['correlation/ordinal/heatmap.pdf'],
    ],
    # Mean cell composition per group, both tasks side by side — the R-drawn counterpart to the EpiSCORE panel of the
    # marker figures, kept separate so both bars can be read at full size.
    'deconvolution': [
        ['deconvolution/stacked_bar.pdf', 'deconvolution/stacked_bar_therapeutic.pdf'],
    ],
    # 8. Classification feature importance: the markers model per feature and by feature block, the stacked model by
    # block only (see the note on figure 3).
    'classification_importance': [
        [f'{_IMPORTANCE}/permutation__model__classification__markers__none__none__random_forest.pdf',
         [f'{_IMPORTANCE}/permutation_grouped__model__classification__markers__none__none__random_forest.pdf',
          f'{_IMPORTANCE}/permutation_grouped__model__classification__stacked__none__none__mlp.pdf']],
    ],
    # 9. Model-vs-model AUROC differences, macro plus one per class.
    'classification_comparison': [
        ['classifier/comparison/comparison__classification__auroc_macro.pdf',
         'classifier/comparison/comparison__classification__auroc_NV.pdf'],
        ['classifier/comparison/comparison__classification__auroc_IM.pdf',
         'classifier/comparison/comparison__classification__auroc_NIM.pdf'],
    ],
    # Cohort overview: the sample-level view of the 500 CpGs most associated with the diagnoses
    'cohort_overview': [
        ['cohort/tsne__classification.pdf', 'cohort/tsne__ordinal.pdf'],
        ['cohort/heatmap_classification_cpgs.pdf'],
    ],
}


def _label_fontfile() -> str:
    """Path to the bold face of the shared figure font, for the panel letters.

    Resolved through matplotlib so the letters come from the same family as the panels' own text; PyMuPDF's built-in
    ``hebo`` is base-14 Helvetica, which is not embedded and renders in whatever the viewer substitutes.
    """
    return font_manager.findfont(
        font_manager.FontProperties(family=list(FIGURE['font_family']), weight='bold')
    )


def assemble(name: str, rows: list[list[str | list[str]]], out_dir: Path) -> Path:
    """Composite one figure's panels onto a single page and write it to ``<out_dir>/<name>.pdf``.

    :param name: Figure name; also the output file stem.
    :param rows: Rows of plot paths relative to :data:`config.PLOTS_DIR`, in panel-letter order. A row entry that is
        itself a list becomes one column of vertically stacked panels within that row.
    :param out_dir: Directory for the assembled figure (created if missing).
    :return: The path written.
    """
    gutter, strip = GUTTER_MM * MM, LABEL_STRIP_MM * MM
    page_w = FIGURE['panel_mm']['full'] * MM

    # Measure everything first: a row's band is set by its tallest column, and the bands set the page height. Every
    # entry is normalised to a column so single panels and stacks take the same placement path below.
    grid = []
    for row in rows:
        columns = []
        for entry in row:
            panels = []
            for rel in [entry] if isinstance(entry, str) else entry:
                path = PLOTS_DIR / rel
                if not path.exists():
                    raise FileNotFoundError(f'Panel {rel!r} of figure {name!r} is missing at {path}. '
                                            f'Re-run the analysis that writes it before assembling.')
                src = pymupdf.open(path)
                panels.append((src, src[0].rect.width, src[0].rect.height))
            width = max(p[1] for p in panels)
            height = sum(strip + p[2] for p in panels) + gutter * (len(panels) - 1)
            columns.append((panels, width, height))
        grid.append(columns)

    bands = [max(c[2] for c in columns) for columns in grid]
    page_h = sum(bands) + gutter * (len(grid) - 1)

    doc = pymupdf.open()
    page = doc.new_page(width=page_w, height=page_h)
    page.insert_font(fontname='panel', fontfile=_label_fontfile())

    letters = iter('abcdefghijklmnopqrstuvwxyz')
    y = 0.0
    for i, (columns, band) in enumerate(zip(grid, bands), start=1):
        row_w = sum(c[1] for c in columns) + gutter * (len(columns) - 1)
        if row_w > page_w:
            raise ValueError(
                f'{name}: row {i} is {row_w / MM:.0f} mm wide but the page is only {page_w / MM:.0f} mm. Move a panel '
                f'to its own row, or draw one of them at a narrower slot.'
            )
        x = (page_w - row_w) / 2  # centre the row; never stretch a panel to fill the width
        for panels, col_w, _ in columns:
            cy = y  # columns are top-aligned, so a short stack leaves its slack at the bottom of the band
            for src, w, h in panels:
                page.show_pdf_page(pymupdf.Rect(x, cy + strip, x + w, cy + strip + h), src, 0)
                page.insert_text((x, cy + strip - 1), next(letters),
                                 fontsize=FIGURE['panel_label_size'], fontname='panel')
                cy += strip + h + gutter
                src.close()
            x += col_w + gutter
        y += band + gutter

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.pdf'
    doc.subset_fonts()  # the label font is inserted whole; the panels' own fonts arrive already subset
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()

    w_mm, h_mm = page_w / MM, page_h / MM
    n = sum(len(panels) for columns in grid for panels, _, _ in columns)
    if h_mm > FIGURE['max_height_mm']:
        # Not rescaled: shrinking to fit would drop the text below the guideline minimum.
        logger.warning('%s: %.0f mm tall exceeds the %d mm page maximum by %.0f mm - regroup its rows or shorten a '
                       'panel upstream rather than scaling the figure down.',
                       name, h_mm, FIGURE['max_height_mm'], h_mm - FIGURE['max_height_mm'])
    logger.info('%s: %d panels, %.0f x %.0f mm -> %s', name, n, w_mm, h_mm, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', choices=sorted(FIGURES), default=None,
                        help='Assemble only this figure (default: all of them).')
    parser.add_argument('--out-dir', type=Path, default=MANUSCRIPT_PLOTS,
                        help=f'Output directory (default: {MANUSCRIPT_PLOTS}).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    names = [args.figure] if args.figure else list(FIGURES)
    for name in names:
        assemble(name, FIGURES[name], args.out_dir)


if __name__ == '__main__':
    main()
