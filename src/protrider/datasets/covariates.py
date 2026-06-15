from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence
import logging


logger = logging.getLogger(__name__)

__all__ = ['parse_covariates']

def parse_covariates(sa_file: Optional[str], cov_used: Optional[list], index_order: Optional[Sequence[str]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Parse covariates from sample annotation file.
    
    Args:
        sa_file: Path to sample annotation file (CSV/TSV)
        cov_used: List of covariate column names to use
        index_order: Order of samples in intensity matrix (pd.Index)
        
    Returns:
        tuple: (covariates, centered_covariates_noNA)
            - covariates: Raw covariates with NAs replaced by 0
            - centered_covariates_noNA: Centered numerical covariates with NAs replaced by 0
    """
    if sa_file is None:
        raise ValueError("Sample annotation file is required.")
    if cov_used is None:
        raise ValueError("Covariates to use must be specified.")

    # Read sample annotation file
    sample_anno = read_annotation_file(sa_file)
    logger.info(f'Finished reading sample annotation with shape: {sample_anno.shape}')
    
    # Sort sa based on intensity sample orders
    if index_order is not None:
        sample_anno = sort_sa_file(sample_anno, index_order)
    
    # Process covariates
    processed_covariates = _process_covariates(sample_anno[cov_used])
    
    # Combine all covariate types
    covariates, centered_covariates = _combine_covariates(processed_covariates)
    
    # Validate output
    assert np.isnan(covariates).sum() == 0, "Covariates contain NaN values"
    assert np.isnan(centered_covariates).sum() == 0, "Centered covariates contain NaN values"
    
    return covariates, centered_covariates


def read_annotation_file(sa_file):
    """Read sample annotation file based on file extension."""
    file_extension = Path(sa_file).suffix
    if file_extension == '.csv':
        return pd.read_csv(sa_file)
    elif file_extension == '.tsv':
        return pd.read_csv(sa_file, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def sort_sa_file(sample_anno: pd.DataFrame, index_order: pd.Index) -> pd.DataFrame:
    """
    Sort a sample annotation DataFrame based on intensity matrix sample order
    
    Args:
        sample_anno : The DataFrame to be sorted. (pd.DataFrame)
        index_order : Order of the intensity matrix indices. (pd.Index)
    
    Returns
    -------
    pd.DataFrame
        The sorted sample_annotation DataFrame.
    
    Raises
    ------
    ValueError
        If the lengths of `sample_anno` and `index_order` do not match,
        or if the indices in `index_order` do not match those in `sample_anno`.
    """
    sample_anno.index = sample_anno.sample_ID
    sample_anno.index.names = ["index"]
    index_list = list(index_order)
      
    # Restrict to only samples in intentisy file
    sample_anno = sample_anno[sample_anno["sample_ID"].isin(index_list)]
    
    # Check length
    if len(sample_anno) != len(index_order):
        raise ValueError(
            f"Length mismatch: sample_anno has {len(sample_anno)} rows, "
            f"but index_order has {len(index_order)} entries."
        )
    
    # Check if all indices match
    if not set(sample_anno.index) == set(index_list):
        raise ValueError("Indices in index_order do not match sample_anno.index.")
    # Create an order DataFrame with explicit positions (keeps duplicates and order)
    order_df = pd.DataFrame({'sample_ID': index_list, '__order_pos': range(len(index_list))})

    # Merge then sort by the position; using inner join because we've already validated multisets
    merged = pd.merge(order_df, sample_anno, on='sample_ID', how='left')

    merged_sorted = merged.sort_values('__order_pos', kind='stable').drop(columns='__order_pos')

    # Set index to the requested order (index_list) so the returned DF index matches index_order exactly
    merged_sorted.index = index_list
     
    return merged_sorted
    

def _is_numeric_dtype(dtype):
    """Check if pandas dtype is numeric."""
    return pd.api.types.is_numeric_dtype(dtype)


def _process_covariates(cov_data):
    """Process covariates into numerical, categorical, and NA indicator variables."""
    numerical_covs = []
    categorical_covs = []
    na_indicator_covs = []
    
    for col_name, series in cov_data.items():
        # Skip covariates with insufficient variation
        if series.nunique() < 2:
            logger.warning(f"Covariate '{col_name}' has less than 2 unique values. Skipping...")
            continue
        
        # Create NA indicator if needed
        if series.isna().any():
            na_indicator = series.isna().astype(int).to_frame(name=f'{col_name}_NA')
            na_indicator_covs.append(na_indicator)
        
        # Process based on data type
        if _is_numeric_dtype(series.dtype) and series.dtype != bool:
            numerical_covs.append(series.to_frame())
        else:
            # One-hot encode categorical variables
            categorical_encoded = pd.get_dummies(series, drop_first=True, dummy_na=False, prefix=col_name)
            categorical_covs.append(categorical_encoded)
    
    # Log covariate counts
    logger.info(f'No. numerical covariates used: {len(numerical_covs)}')
    logger.info(f'No. categorical covariates used: {len(categorical_covs)}')
    logger.info(f'No. NA indicator covariates used: {len(na_indicator_covs)}')
    
    return {
        'numerical': numerical_covs,
        'categorical': categorical_covs,
        'na_indicators': na_indicator_covs
    }


def _combine_covariates(processed_covariates):
    """Combine processed covariates into final arrays."""
    covariate_arrays = []
    centered_covariate_arrays = []
    
    # Process numerical covariates
    if processed_covariates['numerical']:
        numerical_data = pd.concat(processed_covariates['numerical'], axis=1).values
        # Center numerical data and handle NAs
        means = np.nanmean(numerical_data, axis=0, keepdims=True)
        centered_numerical = numerical_data - means
        
        # Replace NAs with 0
        numerical_no_na = np.where(np.isnan(numerical_data), 0, numerical_data)
        centered_numerical_no_na = np.where(np.isnan(centered_numerical), 0, centered_numerical)
        
        covariate_arrays.append(numerical_no_na)
        centered_covariate_arrays.append(centered_numerical_no_na)
    
    # Process categorical covariates (no centering needed)
    if processed_covariates['categorical']:
        categorical_data = pd.concat(processed_covariates['categorical'], axis=1).values
        covariate_arrays.append(categorical_data)
        centered_covariate_arrays.append(categorical_data)
    
    # Process NA indicators (no centering needed)
    if processed_covariates['na_indicators']:
        na_data = pd.concat(processed_covariates['na_indicators'], axis=1).values
        covariate_arrays.append(na_data)
        centered_covariate_arrays.append(na_data)
    
    # Concatenate all covariate types
    if covariate_arrays:
        covariates = np.concatenate(covariate_arrays, axis=1)
        centered_covariates = np.concatenate(centered_covariate_arrays, axis=1)
    else:
        raise ValueError("No valid covariates found.")
    
    return covariates, centered_covariates
