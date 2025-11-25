from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import copy
from pydeseq2.preprocessing import deseq2_norm
import logging

logger = logging.getLogger(__name__)

def read_protein_intensities(input_intensities: str, index_col: str) -> pd.DataFrame:
    """Read protein intensities from a file.
    
    Args:
        input_intensities: Path to file (CSV, TSV, or Parquet)
        index_col: Name of the index column containing protein IDs
    
    Returns:
        pd.DataFrame: Protein intensities with samples as rows and proteins as columns
    """
    file_extension = Path(input_intensities).suffix
    if file_extension == '.csv':
        data = pd.read_csv(input_intensities).set_index(index_col)
    elif file_extension == '.tsv':
        data = pd.read_csv(input_intensities,
                           sep='\t').set_index(index_col)
    elif file_extension == '.parquet':
        data = pd.read_parquet(input_intensities).set_index(index_col)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    data = data.T
    data.index.names = ['sampleID']
    data.columns.name = 'proteinID'
    

    logger.info(f'Finished reading raw data with shape: {data.shape}')

    return data

def preprocess_protein_intensities(data, log_func, maxNA_filter):
    """Preprocess protein intensities data.

    Args:
        data (pd.DataFrame): Input protein intensities data.
        log_func (callable): Function to apply log transformation.
        maxNA_filter (float): Maximum allowed proportion of NA values.

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

        # normalize data with deseq2
        size_factors = None
        deseq_out, size_factors = deseq2_norm(filtered_data.replace(np.nan, 0,
                                                                inplace=False))
        ### check that deseq2 worked, otherwise ignore
        if deseq_out.isna().sum().sum() == 0:
            processed_data = deseq_out
            processed_data.replace(0, np.nan, inplace=True)
            size_factors = size_factors

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