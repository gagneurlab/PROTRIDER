from typing import Iterable, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from numpy._typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import copy
from pydeseq2.preprocessing import deseq2_norm
from abc import ABC
from optht import optht
from tqdm import tqdm
import logging

# Create a logger at the top of the file
logger = logging.getLogger(__name__)


class PCADataset(ABC):
    def __init__(self):
        # centered log data around the protein means
        self.centered_log_data_noNA = None
        # one hot encoding of covariates
        self.cov_one_hot = None
        self.centered_log_data_noNA = None
        self.U = None
        self.s = None
        self.Vt = None

    def perform_svd(self):
        self.U, self.s, self.Vt = np.linalg.svd(np.hstack([self.centered_log_data_noNA,
                                                           self.cov_one_hot.detach().cpu()
                                                           ]),
                                                full_matrices=False)
        logger.info(f'Finished fitting SVD with shapes U: {self.U.shape}, s: {self.s.shape}, Vt: {self.Vt.shape}')

    def find_enc_dim_optht(self):
        try:
            q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None)
        except:
            self.perform_svd()
            q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None)
        return q


class ProtriderDataset(Dataset, PCADataset):
    def __init__(self, csv_file, index_col, sa_file=None,
                 cov_used=None, log_func=np.log,
                 maxNA_filter=0.3, device=torch.device('cpu')):
        super().__init__()

        # read csv
        file_extension = Path(csv_file).suffix
        if file_extension == '.csv':
            self.data = pd.read_csv(csv_file).set_index(index_col)
        elif file_extension == '.tsv':
            self.data = pd.read_csv(csv_file,
                                    sep='\t').set_index(index_col)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        self.device = device
        self.data = self.data.T
        self.data.index.names = ['sampleID']
        self.data.columns.name = 'proteinID'
        logger.info(f'Finished reading raw data with shape: {self.data.shape}')

        # replace 0 with NaN (for proteomics intensities)
        self.data.replace(0, np.nan, inplace=True)

        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(self.data), axis=0)
        self.data = (self.data.T[filtered <= maxNA_filter]).T
        logger.info(
            f"Filtering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {self.data.shape}")
        self.raw_data = copy.deepcopy(self.data)  ## for storing output

        # normalize data with deseq2
        deseq_out, size_factors = deseq2_norm(self.data.replace(np.nan, 0,
                                                                inplace=False))
        ### check that deseq2 worked, otherwise ignore
        if deseq_out.isna().sum().sum() == 0:
            self.data = deseq_out
            self.data.replace(0, np.nan, inplace=True)

        # log data
        self.data = log_func(self.data)
        #### FINISHED PREPROCESSING

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

        # sample annotation including covariates
        if sa_file is not None:
            sa_file_extension = Path(sa_file).suffix
            if sa_file_extension == '.csv':
                sample_anno = pd.read_csv(sa_file)
            elif sa_file_extension == '.tsv':
                sample_anno = pd.read_csv(sa_file, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {sa_file_extension}")
            logger.info(f'Finished reading sample annotation with shape: {sample_anno.shape}')
        else:
            cov_used = None

        if cov_used is not None:
            self.covariates = sample_anno.loc[:, cov_used]
            num_types = ["float64", "float32", "float16",
                         "complex64", "complex128", "int64",
                         "int32", "int16", "int8", "uint8"]
            for col in self.covariates.columns:
                if self.covariates.loc[:, col].dtype not in num_types:
                    self.covariates[col] = pd.factorize(self.covariates[col])[0]
                    self.covariates[col] = np.where(self.covariates[col] < 0,
                                                    np.max(self.covariates[col]) + 1,
                                                    self.covariates[col])
            self.covariates = torch.tensor(self.covariates.values)

            # one_hot encoding of covariates
            for col in range(self.covariates.shape[1]):
                one_hot_col = F.one_hot(self.covariates[:, col], num_classes=self.covariates[:, col].max().numpy() + 1)
                try:
                    one_hot = torch.cat((one_hot, one_hot_col), dim=1)
                except:
                    one_hot = one_hot_col
            self.cov_one_hot = one_hot
        else:
            self.covariates = torch.empty(self.X.shape[0], 0)
            self.cov_one_hot = torch.empty(self.X.shape[0], 0)
        logger.info(f'Finished reading covariates. No. one-hot-encoded covariates used: {self.cov_one_hot.shape[1]}')
        ### Send data to cpu/gpu device
        self.X = self.X.to(device)
        self.torch_mask = self.torch_mask.to(device)
        self.cov_one_hot = self.cov_one_hot.to(device)
        self.prot_means_torch = self.prot_means_torch.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.cov_one_hot[idx], self.prot_means_torch)


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

    @property
    def cov_one_hot(self):
        return self.dataset.cov_one_hot[self.indices]

    @staticmethod
    def concat(subsets: Iterable['ProtriderSubset']):
        """
        Concatenate multiple ProtriderSubset instances into a single one.
        """
        indices = np.concatenate([subset.indices for subset in subsets])
        return ProtriderSubset(subsets[0].dataset, indices)


class ProtriderLOOCVGenerator:
    """
    Cross-validation generator for the ProtriderDataset.
    Creates train, validation, and test splits for k-fold cross validation.
    """

    def __init__(self, input_intensities: str, sample_annotation: str, index_col: str,
                 cov_used: Iterable[str], maxNA_filter: float, log_func: Callable[[ArrayLike], ArrayLike],
                 device=torch.device('cpu')):
        """
        Args:
            input_intensities: Path to CSV file with protein intensity data
            sample_annotation: Path to CSV file with sample annotations
            index_col: Name of the index column
            cov_used: List of covariates to use
            maxNA_filter: Maximum proportion of NAs allowed per protein
            log_func: Log function to apply to the data
            num_folds: Number of cross-validation folds
        """
        self.input_intensities = input_intensities
        self.sample_annotation = sample_annotation
        self.index_col = index_col
        self.cov_used = cov_used
        self.maxNA_filter = maxNA_filter
        self.log_func = log_func

        # Initialize the dataset
        self.dataset = ProtriderDataset(csv_file=input_intensities,
                                        index_col=index_col,
                                        sa_file=sample_annotation,
                                        cov_used=cov_used,
                                        log_func=log_func,
                                        maxNA_filter=maxNA_filter, device=device)

        # Set up LOO
        self.loo = LeaveOneOut()

    def __iter__(self):
        """Generate train, validation, and test subsets for each fold"""
        for train_val_idx, test_idx in tqdm(self.loo.split(self.dataset), total=len(self.dataset)):
            # Further split train_val into train / val
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.2,
                shuffle=True
            )

            # Create subsets
            train_subset = ProtriderSubset(self.dataset, train_idx)
            val_subset = ProtriderSubset(self.dataset, val_idx)
            test_subset = ProtriderSubset(self.dataset, test_idx)

            yield train_subset, val_subset, test_subset

    def __len__(self):
        """Return the number of folds"""
        return len(self.dataset)  # Number of samples in the dataset


class ProtriderKfoldCVGenerator:
    """
    Cross-validation generator for the ProtriderDataset.
    Creates train, validation, and test splits for k-fold cross validation.
    """

    def __init__(self, input_intensities: str, sample_annotation: str, index_col: str,
                 cov_used: Iterable[str], maxNA_filter: float,
                 log_func: Callable[[ArrayLike], ArrayLike], num_folds: int = 5, device=torch.device('cpu')):
        """
        Args:
            input_intensities: Path to CSV file with protein intensity data
            sample_annotation: Path to CSV file with sample annotations
            index_col: Name of the index column
            cov_used: List of covariates to use
            maxNA_filter: Maximum proportion of NAs allowed per protein
            log_func: Log function to apply to the data
            num_folds: Number of cross-validation folds
        """
        self.input_intensities = input_intensities
        self.sample_annotation = sample_annotation
        self.index_col = index_col
        self.cov_used = cov_used
        self.maxNA_filter = maxNA_filter
        self.log_func = log_func
        self.num_folds = num_folds

        # Initialize the dataset
        self.dataset = ProtriderDataset(csv_file=input_intensities,
                                        index_col=index_col,
                                        sa_file=sample_annotation,
                                        cov_used=cov_used,
                                        log_func=log_func,
                                        maxNA_filter=maxNA_filter,
                                        device=device)

        # Set up KFold
        self.kf = KFold(n_splits=num_folds, shuffle=True)

        # Pre-compute all folds for consistency
        self._folds = list(self.kf.split(self.dataset))

    def __iter__(self):
        for run_idx in tqdm(range(self.num_folds)):
            test_idx = run_idx
            pca_idx = (run_idx + 1) % self.num_folds
            val_idx = (run_idx + 2) % self.num_folds
            train_idx = [i for i in range(self.num_folds) if i not in [test_idx, pca_idx, val_idx]]

            # indices for each part
            test_indices = self._folds[test_idx][1]
            pca_indices = self._folds[pca_idx][1]
            val_indices = self._folds[val_idx][1]
            train_indices = np.concatenate([self._folds[i][1] for i in train_idx])

            # Create subsets
            train_subset = ProtriderSubset(self.dataset, train_indices)
            pca_subset = ProtriderSubset(self.dataset, pca_indices)
            val_subset = ProtriderSubset(self.dataset, val_indices)
            test_subset = ProtriderSubset(self.dataset, test_indices)

            yield pca_subset, train_subset, val_subset, test_subset

    def __len__(self):
        """Return the number of folds"""
        return self.num_folds
