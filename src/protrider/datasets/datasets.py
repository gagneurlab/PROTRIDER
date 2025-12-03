from __future__ import annotations

from typing import Iterable, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import copy
from abc import ABC
from optht import optht
import logging
from .covariates import parse_covariates
from .protein_intensities import read_protein_intensities, preprocess_protein_intensities

logger = logging.getLogger(__name__)

class PCADataset(ABC):
    def __init__(self):
        # centered log data around the protein means
        self.centered_log_data_noNA = None
        self.U = None
        self.s = None
        self.Vt = None

    def perform_svd(self):
        self.U, self.s, self.Vt = np.linalg.svd(self.centered_log_data_noNA, full_matrices=False)
        logger.info(f'Finished fitting SVD with shapes U: {self.U.shape}, s: {self.s.shape}, Vt: {self.Vt.shape}')

    def find_enc_dim_optht(self):
        if self.s is None:
            self.perform_svd()

        q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None)
        return q


class ProtriderDataset(Dataset, PCADataset):
    def __init__(self, input_intensities: str, index_col: str, 
                 sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None, log_func: Callable = np.log,
                 maxNA_filter: float = 0.3, device: torch.device = torch.device('cpu'),
                 input_format: str = "proteins_as_rows"):
        """Initialize ProtriderDataset.
        
        Args:
            input_intensities: Path to protein intensities file (CSV, TSV, or Parquet)
            index_col: Name of the index column containing protein IDs
            sa_file: Path to sample annotations file (CSV/TSV), or None
                    - Format: rows = samples
            cov_used: List of covariate column names to use, or None
            log_func: Function to apply log transformation (default: np.log)
            maxNA_filter: Maximum allowed proportion of NA values (default: 0.3)
            device: PyTorch device (default: 'cpu')
            input_format: Format of input file:
                         - "proteins_as_rows": proteins are rows, samples are columns (default)
                         - "proteins_as_columns": samples are rows, proteins are columns
        """
        super().__init__()
        self.device = device

        # Read and preprocess protein intensities
        unfiltered_data = read_protein_intensities(input_intensities, index_col, input_format)
        self.data, self.raw_data, self.size_factors = preprocess_protein_intensities(
            unfiltered_data, log_func, maxNA_filter
        )

        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used)
                self.covariates = torch.from_numpy(self.covariates)
                self.centered_covariates_noNA = torch.from_numpy(self.centered_covariates_noNA)
            except ValueError:
                logger.warning("No valid covariates found after parsing.")
                self.covariates = torch.empty(self.data.shape[0], 0)
                self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)
        else:
            self.covariates = torch.empty(self.data.shape[0], 0)
            self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)

        
        # store protein means
        self.prot_means = np.nanmean(self.data, axis=0, keepdims=1)

        ## Center and mask NaNs in input
        self.mask = ~np.isfinite(self.data)
        self.centered_log_data_noNA = self.data - self.prot_means
        self.centered_log_data_noNA = np.where(self.mask, 0, self.centered_log_data_noNA)

        # Input and output of autoencoder is:
        # uncentered data without NaNs, replacing NANs with means
        self.X = self.centered_log_data_noNA + self.prot_means  ## same as data but without NAs

        ## to torch
        self.X = torch.tensor(self.X)
        # self.X_target = self.X ### needed for outlier injection
        self.mask = np.array(self.mask.values)
        self.torch_mask = torch.tensor(self.mask)
        self.prot_means_torch = torch.from_numpy(self.prot_means).squeeze(0)

        ### Send data to cpu/gpu device
        self.X = self.X.to(device)
        self.torch_mask = self.torch_mask.to(device)
        self.covariates = self.covariates.to(device)
        self.prot_means_torch = self.prot_means_torch.to(device)
        # self.presence = (~self.torch_mask).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.covariates[idx], self.prot_means_torch)


class ProtriderSubset(Subset, PCADataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.prot_means = np.nanmean(self.data, axis=0, keepdims=1)
        self.prot_means_torch = torch.from_numpy(self.prot_means).squeeze(0).to(dataset.device)

    @property
    def X(self):
        return self.dataset.X[self.indices]

    @property
    def data(self):
        return self.dataset.data.iloc[self.indices]

    @property
    def raw_data(self):
        return self.dataset.raw_data.iloc[self.indices]

    @property
    def mask(self):
        return self.dataset.mask[self.indices]

    @property
    def torch_mask(self):
        return self.dataset.torch_mask[self.indices]

    @property
    def centered_log_data_noNA(self):
        return self.dataset.centered_log_data_noNA[self.indices]

    @property
    def covariates(self):
        return self.dataset.covariates[self.indices]

    @staticmethod
    def concat(subsets: Iterable['ProtriderSubset']):
        """
        Concatenate multiple ProtriderSubset instances into a single one.
        """
        indices = np.concatenate([subset.indices for subset in subsets])
        return ProtriderSubset(subsets[0].dataset, indices)

    def deepcopy_to_dataset(self) -> Dataset:
        """
        Convert the ProtriderSubset instance back to a ProtriderDataset instance.
        """
        dataset = copy.deepcopy(self.dataset)
        dataset.X = self.X
        dataset.data = self.data
        dataset.raw_data = self.raw_data
        dataset.mask = self.mask
        dataset.torch_mask = self.torch_mask
        dataset.centered_log_data_noNA = self.centered_log_data_noNA
        dataset.covariates = self.covariates
        dataset.prot_means = self.prot_means
        dataset.prot_means_torch = self.prot_means_torch
        return dataset