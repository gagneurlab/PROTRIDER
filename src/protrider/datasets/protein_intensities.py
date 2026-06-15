from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import copy
from pydeseq2.preprocessing import deseq2_norm
import logging

logger = logging.getLogger(__name__)


def read_protein_intensities(input_intensities, index_col: str, input_format: str = "proteins_as_rows") -> pd.DataFrame:
    """Read protein intensities from one or more files.

    Args:
        input_intensities: Path to a file, or a list of paths whose columns are
                           concatenated (CSV, TSV, or Parquet; optionally .gz)
        index_col: Name of the index column containing protein IDs
        input_format: Format of the input file:
                     - "proteins_as_rows": proteins are rows, samples are columns (default)
                     - "proteins_as_columns": samples are rows, proteins are columns

    Returns:
        pd.DataFrame: Protein intensities with samples as rows and proteins as columns
    """
    # Accept a single path (str/Path) or a list of paths.
    if isinstance(input_intensities, (str, Path)):
        input_intensities = [input_intensities]
    intensities = []
    for input_intensity in input_intensities:
        suffixes = Path(input_intensity).suffixes
        compression = None
        if suffixes[-1] == '.gz':
            compression = 'gzip'
            suffixes = suffixes[:-1]
        if suffixes[-1] == '.csv':
            temp_data = pd.read_csv(input_intensity, compression=compression).set_index(index_col)
        elif suffixes[-1] == '.tsv':
            temp_data = pd.read_csv(input_intensity, sep='\t', compression=compression).set_index(index_col)
        elif suffixes[-1] == '.parquet':
            temp_data = pd.read_parquet(input_intensity, engine='fastparquet').set_index(index_col)
        else:
            raise ValueError(f"Unsupported file type: {suffixes[-1]}")
        intensities.append(temp_data)
    data = pd.concat(intensities, axis=1)
    
    # Transpose if needed to get samples as rows, proteins as columns
    if input_format == "proteins_as_rows":
        data = data.T
        data.index.names = ['sampleID']
        data.columns.name = 'proteinID'
    elif input_format == "proteins_as_columns":
        # Already in the correct format, just set names
        data.index.name = 'sampleID'
        data.columns.name = 'proteinID'
    else:
        raise ValueError(f"Invalid input_format: {input_format}. Must be 'proteins_as_rows' or 'proteins_as_columns'")
    
    logger.info(f'Finished reading raw data with shape: {data.shape} (samples x proteins)')

    return data

def preprocess_protein_intensities(data, log_func, maxNA_filter, normalize=True):
    """Preprocess protein intensities data.

    Args:
        data (pd.DataFrame): Input protein intensities data.
        log_func (callable): Function to apply log transformation.
        maxNA_filter (float): Maximum allowed proportion of NA values.
        normalize (bool): Whether to apply DESeq2 size-factor normalization
            before log transformation. Only applies when
            ``log_func`` is not None.

    Returns:
        tuple: Processed protein intensities, filtered (no NAs) raw data, and size factors.
    """

    raw_data = None
    size_factors = None
    processed_data = data
    if log_func is not None:
        # replace 0 with NaN (for proteomics intensities)
        processed_data.replace(0, np.nan, inplace=True)

        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(processed_data), axis=0)
        filtered_data = (processed_data.T[filtered <= maxNA_filter]).T
        logger.info(
            f"Filtering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {data.shape}")
        raw_data = copy.deepcopy(processed_data)  ## for storing output

        if normalize:
            # normalize data with deseq2
            deseq_out, size_factors = deseq2_norm(filtered_data.replace(np.nan, 0,
                                                                    inplace=False))
            ### check that deseq2 worked, otherwise ignore
            if deseq_out.isna().sum().sum() == 0:
                processed_data = deseq_out
                processed_data.replace(0, np.nan, inplace=True)
            else:
                size_factors = None
        else:
            logger.info("Skipping DESeq2 size-factor normalization.")
            processed_data = filtered_data

        # log data
        processed_data = log_func(processed_data)
    else:
        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(processed_data), axis=0)
        processed_data = (processed_data.T[filtered <= maxNA_filter]).T
        logger.info(
            f"Filtering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {processed_data.shape}")
        raw_data = copy.deepcopy(processed_data)  ## for storing output

    return processed_data, raw_data, size_factors
