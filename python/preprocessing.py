import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import _CONFIG, _cfg_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)


def clean_sample_map(output_path: Path, detection_rate: Optional[float] = None) -> None:
    """
    Outputs a cleaned version of the methylation sample map CSV.
    Removes bad samples, fixes slideId typos/formatting, and adds a sentrix_key column
    so that downstream R scripts can use this file directly without duplicating cleaning logic.

    :param output_path: Path to write the cleaned sample map CSV.
    :param detection_rate: Optional minimum detection rate threshold for quality filtering.
    :return: None
    """
    logger.info(f'Cleaning sample map -> {output_path.as_posix()}')

    df = pd.read_csv(_cfg_path('methylation_samples'), dtype=object)
    df['sentrix_key'] = df['Sentrix_ID'] + '_' + df['Sentrix_Position']

    # drop sample-sheet entries without an actual array (no sentrix_key); they have a slideId but were never measured.
    df = df[df['sentrix_key'].notna()]

    # remove bad samples
    drop_keys = set(_samples_to_drop(detection_rate))
    df = df[~df['sentrix_key'].isin(drop_keys)]

    # clean slideIds
    df = _clean_sample_ids(df)

    df.to_csv(output_path, index=False)
    logger.info(f'Cleaned sample map: {len(df)} samples')


def combine_betas(directory: Path, output_path: Path, manifest_path: Path, detection_rate: Optional[float] = None) -> None:
    """
    Combines beta values from multiple CSV files located in the specified directory into a single DataFrame, processes
    the data (cleaning, harmonizing names, adding genome coordinates), and saves the result to the specified output path.

    :param directory: Path object representing the directory containing the CSV files with beta values to be combined.
    :param output_path: Path object specifying the location where the combined and processed beta values DataFrame
        should be saved as a CSV file.
    :param manifest_path: Path object specifying the manifest file path that will be used to extract genome coordinates.
    :param detection_rate: Optional float value specifying the minimum acceptable detection rate for quality control.
        Samples failing this threshold will be removed during processing.
    :return: None
    """
    logger.info(f'Combining beta values from folder: {directory.as_posix()} into {output_path.as_posix()}')

    logger.debug('Merging beta values into one dataframe...')
    combined = pd.DataFrame()
    for file in directory.rglob("*betas*.csv"):
        batch = pd.read_csv(file, index_col=0)

        combined = combined.join(batch, how="outer")

    logger.debug(f'Removing samples (missing SlideId, non SCP ones, insufficient quality (<{detection_rate}), ...)')
    combined = _drop_samples(combined, detection_rate=detection_rate)

    logger.debug('Converting names...')
    combined.columns = _match_slides(combined.columns)

    logger.debug('Cleaning slideIds...')
    combined.columns = _clean_slide_id_series(combined.columns)

    logger.debug('Adding genome coordinates...')
    combined = combined.join(_get_genome_coordinates(manifest_path))

    logger.debug(f'Writing {output_path.as_posix()}...')
    combined.to_csv(output_path)
    logger.info('Finished.')


def _samples_to_drop(detection_rate: Optional[float] = None) -> list[str]:
    """
    Returns the list of sentrix keys that should be excluded from analyses.

    :param detection_rate: Optional minimum detection rate threshold. Samples below this are included in the drop list.
    :return: List of sentrix key strings to drop.
    """
    # non SCP ones
    to_drop = ['207700170092_R07C01', '207700170092_R08C01', '207700170093_R01C01', '207700170093_R06C01',
               '207700170093_R07C01', '207700170093_R08C01']

    # ones without a slideId
    to_drop += ['207558890010_R01C01', '207558890010_R02C01', '207558890010_R03C01']

    # duplicate (v1 & v2), drop v1 one
    to_drop += ['206601460002_R01C01']

    # if quality is specified, drop all samples that have a detection rate < quality
    if detection_rate:
        epicv1_detectionrate = pd.read_csv(_cfg_path('epicv1_detectionrate'), index_col=0)
        to_drop += epicv1_detectionrate[epicv1_detectionrate.iloc[:, 0] < detection_rate].index.tolist()

        epicv2_detectionrate = pd.read_csv(_cfg_path('epicv2_detectionrate'), index_col=0)
        to_drop += epicv2_detectionrate[epicv2_detectionrate.iloc[:, 0] < detection_rate].index.tolist()

    return to_drop


def _drop_samples(df: pd.DataFrame, detection_rate: Optional[float] = None) -> pd.DataFrame:
    """
    Drops specific samples from the given dataframe based on predefined criteria and an optional detection rate
    threshold. Samples are dropped if they are included in predefined lists, are duplicates, or have a detection
    rate below the specified threshold (if provided).

    :param df: The dataframe from which samples should be dropped.
    :param detection_rate: An optional float specifying the minimum detection rate threshold. Samples with a detection
        rate below this threshold will be dropped.
    :return: A dataframe with the specified samples removed.
    """
    return df.drop(_samples_to_drop(detection_rate), axis=1, errors='ignore')


def _match_slides(names: pd.Series) -> pd.Series:
    """
    Matches a series of sample identifiers to corresponding slide IDs. Retrieves a mapping file that associates Sentrix_ID
    and Sentrix_Position with slide IDs and uses it to map the provided sample identifiers to their respective slide IDs.

    :param names: A pandas Series containing sample identifiers in the format of Sentrix_ID + '_' + Sentrix_Position.
        These identifiers are used to match against the mapping file.
    :return: A pandas Series containing the corresponding slide IDs for the given sample identifiers.
    """
    mapping = pd.read_csv(_cfg_path('methylation_samples'), dtype=object)
    mapping.index = mapping['Sentrix_ID'] + '_' + mapping['Sentrix_Position']
    return mapping.loc[names, 'slideId']


def _get_genome_coordinates(manifest_path: Path) -> pd.Series:
    """
    Extract genome coordinates (CpG_chrm-CpG_beg) from a manifest file based on probe IDs.

    :param manifest_path: A Path object pointing to the manifest file. The file must be in a tab-delimited format with
        at least the columns 'Probe_ID', 'CpG_chrm', and 'CpG_beg'. 'CpG_beg' must contain integer-like values.
    :return: A pandas Series containing formatted genome coordinates. Each entry corresponds to a probe, with its index
        being the 'Probe_ID' and the value as a string representation of its genomic coordinates in the format
        'chromosome-start_position'.
    """
    manifest = pd.read_csv(manifest_path, index_col='Probe_ID', sep='\t', dtype={"CpG_beg": "Int64"})
    return manifest[['CpG_chrm', 'CpG_beg']].astype(str).fillna('nan').agg('-'.join, axis=1).rename('genome_coordinates')


def combine_platforms(betas_v1: Path, betas_v2: Path, output_path: Path, na_frac: float = 0.2) -> None:
    """
    Combines beta values from two platforms into a single dataset based on genomic coordinates after cleaning.

    :param betas_v1: Path to the first beta values file.
    :param betas_v2: Path to the second beta values file.
    :param output_path: Path to the output file where the combined beta values will be stored.
    :param na_frac: Fraction threshold of missing values for filtering out probes.
    :return: None
    """
    logger.info('Combining beta values from %s and %s into %s.',
                betas_v1.as_posix(), betas_v2.as_posix(), output_path.as_posix())

    logger.debug('Cleaning probes...')
    betas_v1 = _get_cleaned_betas(betas_v1, na_frac=na_frac)
    betas_v2 = _get_cleaned_betas(betas_v2, na_frac=na_frac)

    logger.debug('Combining beta values based on genomic coordinates. Shape before: EpicV1 %s, EpicV2 %s...',
                 betas_v1.shape, betas_v2.shape)
    combined = betas_v1.join(betas_v2, how='inner', rsuffix='_v2')
    logger.debug(f'Shape after: {combined.shape}')

    logger.debug(f'Writing combined beta values to {output_path.as_posix()}...')
    combined.to_csv(output_path)


def _get_cleaned_betas(path: Path, na_frac: float = 0.2) -> pd.DataFrame:
    """
    Processes and cleans a beta-value DataFrame by applying several filtering and aggregation steps.

    This function performs the following steps to clean beta values:
    1. Removes probes binding to no site.
    2. Drops probes with a fraction of NaN values equal to or exceeding a specified threshold.
    3. Handles duplicate genome positions by averaging the values and combining corresponding CpG names.
    4. Excludes probes on sex chromosomes (chrX and chrY).

    :param path: The file path to a CSV file containing beta values with 'genome_coordinates' as a column.
    :param na_frac: The threshold for NaN value fraction; probes with a higher fraction will be removed (default: 0.2).
    :return: A cleaned pandas DataFrame indexed by genome coordinates.
    """
    df = pd.read_csv(path, index_col=0)
    # remove probes that bind to no site
    df = df[~df['genome_coordinates'].str.startswith('nan')]

    # remove probes with nans frac >= na_frac
    df = df.dropna(thresh=int(df.shape[1] * (1 - na_frac)), subset=[c for c in df.columns if c != 'genome_coordinates'])

    # Probes sharing a genome position are averaged; their names are kept as a comma-separated Probe_ID.
    probe_id = df.groupby('genome_coordinates').apply(lambda row: ','.join(row.index), include_groups=False)
    df = df.groupby('genome_coordinates').mean()
    df = df.join(probe_id.rename('Probe_ID'))

    # remove sex probes
    df = df[~(df.index.str.startswith('chrX') | df.index.str.startswith('chrY'))]

    return df


def split_train_test(in_path: Path, out_path_train: Path, out_path_test: Path) -> None:
    """
    Splits a dataset into training and testing subsets after cleaning/standardizing sample IDs.

    :param in_path: Path to the input file containing the dataset in CSV format.
    :param out_path_train: Path to the output file where the training data will be saved.
    :param out_path_test: Path to the output file where the testing data will be saved.
    :return: None
    """
    logger.info('Splitting train/test from %s into %s and %s.',
                in_path.as_posix(), out_path_train.as_posix(), out_path_test.as_posix())
    logger.debug('Splitting train/test...')
    train, test = _train_test_split()
    logger.debug(f'Size of train/test: {len(train)}/{len(test)}')

    # slideIds already cleaned in combine_betas()
    betas = pd.read_csv(in_path, index_col=0)

    logger.debug(f'Selecting train/test. Initial size: {betas.shape}...')
    betas_train = betas.reindex(columns=train['slideId']).dropna(axis=1, how='all')
    betas_test = betas.reindex(columns=test['slideId']).dropna(axis=1, how='all')
    logger.debug(f'Shape of train/test: {betas_train.shape}/{betas_test.shape}')

    logger.debug(f'Writing {out_path_train.as_posix()}...')
    betas_train.to_csv(out_path_train)
    logger.debug(f'Writing {out_path_test.as_posix()}...')
    betas_test.to_csv(out_path_test)


def _clean_slide_id_series(ids: pd.Index | pd.Series) -> pd.Index | pd.Series:
    """
    Applies string-level cleaning fixes to a Series or Index of slideId strings.

    :param ids: A pandas Series or Index of slideId strings.
    :return: Cleaned Series or Index with standardized slideId values.
    """
    s = ids.to_series() if isinstance(ids, pd.Index) else ids
    s = s.apply(lambda x: x.replace('_', '-'))
    # Typos in the sample sheet.
    s = s.replace({'580116': '580110', '399629': '399625', '426577-2': '426557-2', '476740-2': '476790-2'})
    # Slides whose -1/-2 lesion suffix is missing.
    s = s.replace({'642345': '642345-1', '199254': '199254-1', '619807': '619807-1', '599560': '599560-1',
                   '109042': '109042-1', '438430': '438430-1', '729980': '729980-2'})
    # Drop trailing non-numeric annotations (e.g. '-A', '-BCD').
    s = s.apply(lambda x: re.sub('-?[^0-9]*$', '', x))
    if isinstance(ids, pd.Index):
        return pd.Index(s)
    return s


def _clean_sample_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes slide IDs in the given DataFrame. Includes fixes for slideIDs

    :param df: A pandas DataFrame containing a column named 'slideId' which needs cleaning and standardization.
    :return: A cleaned and deduplicated DataFrame with standardized 'slideId' values.
    """
    df['slideId'] = _clean_slide_id_series(df['slideId'])
    df = df.dropna(how='all', subset=['slideId'])
    df = df.drop_duplicates(subset=['slideId'])

    return df


def _train_test_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the methylation sample and metadata into training and testing datasets based on a predefined test clinic.
    Harmonizes/cleans slideIDs before merging with metadata. 'Other' diagnoses are removed.

    :returns: A tuple containing two pandas DataFrames:
              - The training dataset (metadata)
              - The testing dataset (metadata)
    """
    sample_sheet = pd.read_csv(_cfg_path('methylation_samples'), dtype=object)
    meta = pd.read_csv(_cfg_path('meta_data'))

    sample_sheet = _clean_sample_ids(sample_sheet)
    full = sample_sheet.merge(meta, on='slideId', how='inner')
    full = full[full['groupedPrimaryDiagnosisPatho'] != 'other']

    # test_clinic is either one clinic or a list of them.
    test_clinics = _CONFIG['test_clinic']
    if isinstance(test_clinics, str):
        test_clinics = [test_clinics]
    train = full[~full['clinic'].isin(test_clinics)]
    test = full[full['clinic'].isin(test_clinics)]

    return train, test


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    clean_sample_map(_cfg_path('methylation_samples_cleaned'), detection_rate=0.9)
    combine_betas(_cfg_path('epic_v1_dir'), _cfg_path('betas_v1'), manifest_path=_cfg_path('manifest_v1'), detection_rate=0.9)
    combine_betas(_cfg_path('epic_v2_dir'), _cfg_path('betas_v2'), manifest_path=_cfg_path('manifest_v2'), detection_rate=0.9)
    combine_platforms(_cfg_path('betas_v1'), _cfg_path('betas_v2'), _cfg_path('betas_combined'))
    split_train_test(_cfg_path('betas_combined'), _cfg_path('betas_train'), _cfg_path('betas_test'))
