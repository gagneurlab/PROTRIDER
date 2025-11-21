from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import copy
from typing import Union
from pydeseq2.preprocessing import deseq2_norm
import logging

logger = logging.getLogger(__name__)

def read_protein_intensities(input_intensities: Union[str, pd.DataFrame], index_col: str) -> pd.DataFrame:
    """Read protein intensities from a file or DataFrame.
    
    Args:
        input_intensities: Path to file (str) or pandas DataFrame with proteins as columns
        index_col: Name of the index column (used only if input is a file path)
    
    Returns:
        pd.DataFrame: Protein intensities with samples as rows and proteins as columns
    """
    # If already a DataFrame, just validate and return
    if isinstance(input_intensities, pd.DataFrame):
        data = input_intensities.copy()
        # Ensure proper structure
        if data.index.name is None:
            data.index.name = 'sampleID'
        if data.columns.name is None:
            data.columns.name = 'proteinID'
        logger.info(f'Using provided DataFrame with shape: {data.shape}')
        return data
    
    # Otherwise, read from file
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