
from __future__ import annotations

from typing import Iterable, Callable
import torch
from numpy._typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from .datasets import ProtriderDataset, ProtriderSubset

logger = logging.getLogger(__name__)

class ProtriderCVGenerator(ABC):
    """
    Cross-validation generator for the ProtriderDataset.
    """

    def __init__(self, input_intensities: str, sample_annotation: str, index_col: str,
                 cov_used: Iterable[str], maxNA_filter: float,
                 log_func: Callable[[ArrayLike], ArrayLike], device=torch.device('cpu')):
        """
        Args:
            input_intensities: Path to CSV file with protein intensity data
            sample_annotation: Path to CSV file with sample annotations
            index_col: Name of the index column
            cov_used: List of covariates to use
            maxNA_filter: Maximum proportion of NAs allowed per protein
            log_func: Log function to apply to the data
        """
        self.input_intensities = input_intensities
        self.sample_annotation = sample_annotation
        self.index_col = index_col
        self.cov_used = cov_used
        self.maxNA_filter = maxNA_filter
        self.log_func = log_func

        # Initialize the dataset
        self.dataset = ProtriderDataset(input_intensities=input_intensities,
                                        index_col=index_col,
                                        sa_file=sample_annotation,
                                        cov_used=cov_used,
                                        log_func=log_func,
                                        maxNA_filter=maxNA_filter,
                                        device=device)

    @abstractmethod
    def _get_splits(self):
        """Generate train, validation, and test subsets for each fold"""
        raise NotImplementedError("Subclasses should implement this method.")

    def __iter__(self):
        for train_val_idx, test_idx in tqdm(self._get_splits(), total=len(self)):
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

    @abstractmethod
    def __len__(self):
        """Return the number of folds"""
        raise NotImplementedError("Subclasses should implement this method.")


class ProtriderLOOCVGenerator(ProtriderCVGenerator):
    """
    Cross-validation generator for the ProtriderDataset.
    Creates train, validation, and test splits for leave-one-out cross validation.
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
        super().__init__(input_intensities, sample_annotation, index_col, cov_used, maxNA_filter, log_func, device)

        # Set up LOO
        self.loo = LeaveOneOut()

    def _get_splits(self):
        return self.loo.split(self.dataset)

    def __len__(self):
        """Return the number of folds"""
        return len(self.dataset)  # Number of samples in the dataset


class ProtriderKfoldCVGenerator(ProtriderCVGenerator):
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
        super().__init__(input_intensities, sample_annotation, index_col, cov_used, maxNA_filter, log_func, device)
        self.num_folds = num_folds
        # Set up KFold
        self.kf = KFold(n_splits=num_folds, shuffle=True)

    def _get_splits(self):
        return self.kf.split(self.dataset)

    def __len__(self):
        """Return the number of folds"""
        return self.num_folds
