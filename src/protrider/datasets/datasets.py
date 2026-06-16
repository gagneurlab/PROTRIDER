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
from pydeseq2.preprocessing import deseq2_norm
import pandas as pd
import numpy as np
import re
from pathlib import Path
import torch.nn.functional as F


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

        if q < 2:
            logger.warning(f"Optimal latent space dimension is smaller than 2. Check your count matrix and"
                             "verify that all samples have the expected number of counts\nFor now, the latent space dimension is set to 2.")
            q = 2
        return q


class ProtriderDataset(Dataset, PCADataset):
    def __init__(self, input_intensities: str, index_col: str, 
                 sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None, log_func: Callable = np.log,
                 maxNA_filter: float = 0.3, device: torch.device = torch.device('cpu'),
                 input_format: str = "proteins_as_rows", normalize: bool = True):
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
            normalize: Whether to apply DESeq2 size-factor normalization before
                      log transformation (default: True).
        """
        super().__init__()
        self.device = device

        # Read and preprocess protein intensities
        unfiltered_data = read_protein_intensities(input_intensities, index_col, input_format)
        self.data, self.raw_data, self.size_factors = preprocess_protein_intensities(
            unfiltered_data, log_func, maxNA_filter, normalize=normalize
        )

        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used, self.data.index)
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
        self.raw_filtered = pd.DataFrame()

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


class OutriderDataset(Dataset, PCADataset):
    def __init__(self, input_intensities: str, index_col: str,
                 sa_file: Optional[str] = None,
                 cov_used: Optional[list] = None, log_func: Callable = np.log,
                 fpkm_cutoff: int = 1, gtf: str = None,
                 device: torch.device = torch.device('cpu'),
                 input_format: str = "proteins_as_rows"):
        super().__init__()

        # Read and preprocess protein intensities
        self.data = read_protein_intensities(input_intensities, index_col)
        # self.data, self.raw_data, self.size_factors = preprocess_protein_intensities(
        #     unfiltered_data, log_func, maxNA_filter
        # )

        self.device = device
        self.data.index.names = ['sampleID']
        self.data.columns.name = 'proteinID'
        self.gtf = gtf
        logger.info(f'Finished reading raw data with shape: {self.data.shape}')

        self.raw_data = copy.deepcopy(self.data)  ## for storing output

        # normalize data
        _, size_factors = deseq2_norm(self.data.replace(np.nan, 0))
        self.size_factors = size_factors[:, np.newaxis]


        # filter based on fpkm
        logger.info(f'Calculating fpkms')

        gene_lengths_df = self.gene_exonic_lengths_from_gtf(self.gtf)
        self.fpkms = self.calculate_fpkm(self.raw_data.T, gene_lengths_df)

        logger.info(f'Filtering based on fpkm')
        _, self.passed_filter = self.filter_genes_by_fpkm(self.fpkms.T, fpkm_cutoff=fpkm_cutoff, percentage=0.05)
        # passed_filter is indexed by gene and may cover fewer genes than the data: genes
        # without an exonic length in the GTF get no fpkm and so are absent from the filter.
        # Align the mask to the data's columns by LABEL (missing genes -> dropped) instead of
        # applying it positionally, which fails when len(passed_filter) != data.shape[1].
        self.passed_filter = self.passed_filter.reindex(self.data.columns, fill_value=False)
        self.data = self.data.loc[:, self.passed_filter.values]

        self.raw_filtered = copy.deepcopy(self.data)  ## for storing output/plotting
        # normalize with size_factors
        self.data = np.log((1 + self.data) / self.size_factors)


        #### FINISHED PREPROCESSING

        # store protein means
        self.prot_means = np.nanmean(self.data, axis=0, keepdims=1)

        ## Center and mask NaNs in input
        self.mask = ~np.isfinite(self.data)
        self.centered_log_data_noNA = self.data - self.prot_means
        self.centered_log_data_noNA = np.where(self.mask, 0, self.centered_log_data_noNA)

        prot_means_df = pd.DataFrame({"xbar": np.ravel(self.prot_means), "geneID": self.data.columns})
        # Input and output of autoencoder is:
        # uncentered data without NaNs, replacing NANs with means
        # self.X = self.centered_log_data_noNA + self.prot_means  ## same as data but without NAs
        self.X = self.centered_log_data_noNA

        ## to torch
        self.X = torch.tensor(self.X)
        self.raw_x = torch.tensor(self.raw_filtered.values)

        self.mask = np.array(self.mask.values)
        self.torch_mask = torch.tensor(self.mask)
        self.prot_means_torch = torch.from_numpy(self.prot_means).squeeze(0)

        # Read and preprocess covariates
        if sa_file is not None and cov_used is not None:
            try:
                self.covariates, self.centered_covariates_noNA = parse_covariates(sa_file, cov_used, self.data.index)
                self.covariates = torch.from_numpy(self.covariates)
                self.centered_covariates_noNA = torch.from_numpy(self.centered_covariates_noNA)
            except ValueError as e:
                print(e)
                logger.warning("No valid covariates found after parsing.")
                self.covariates = torch.empty(self.data.shape[0], 0)
                self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)
        else:
            self.covariates = torch.empty(self.data.shape[0], 0)
            self.centered_covariates_noNA = torch.empty(self.data.shape[0], 0)

        ### Send data to cpu/gpu device
        self.X = self.X.to(device)
        self.torch_mask = self.torch_mask.to(device)
        self.covariates = self.covariates.to(device)
        self.prot_means_torch = self.prot_means_torch.to(device)
        self.raw_x = self.raw_x.to(device)
        self.size_factors = torch.tensor(self.size_factors).to(device)
        # self.presence = (~self.torch_mask).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.covariates[idx], self.prot_means_torch, self.raw_x[idx], self.size_factors[idx])

    def filter_genes_by_fpkm(self, fpkm_matrix, fpkm_cutoff=1, percentage=0.05):
        """
        Filters genes based on minimum FPKM in at least 5% of samples.

        Parameters:
            fpkm_matrix (pd.DataFrame): samples × genes (rows are samples, columns are genes)
            min_fpkm (float): Minimum FPKM threshold
            percentage (float): Fraction of samples that must pass the threshold

        Returns:
            passed_filter (pd.Series): Boolean mask for each gene (index = gene names)
            filtered_matrix (pd.DataFrame): Matrix after filtering (samples × genes)
        """
        # Number of samples (rows)
        n_samples = fpkm_matrix.shape[0]

        # Minimum number of samples a gene must pass the threshold in
        min_samples = max(1.0, percentage * n_samples)

        # Boolean mask: for each gene (column), count how many samples (rows) have FPKM ≥ min_fpkm
        passed_filter = (fpkm_matrix > fpkm_cutoff).sum(axis=0) >= min_samples

        logger.info(f"{len(passed_filter) - passed_filter.sum()} genes out of {len(passed_filter)} are filtered out. New shape: ({n_samples}, {passed_filter.sum()})")

        # Apply filter to keep only columns (genes) that passed
        filtered_matrix = fpkm_matrix.loc[:, passed_filter]

        return filtered_matrix, passed_filter

    def gene_exonic_lengths_from_gtf(self, gtf_path: str) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: gene_id, exonic_length
        Collapses overlapping exons per gene across all transcripts --> not necessarily 
        mane_transcript gene length
        """
        def extract_gene_id(attr):
            m = re.search(r'gene_id\s+"([^"]+)"', attr)
            if m:
                return m.group(1)
            m = re.search(r'gene_id=([^;]+)', attr)
            if m:
                return m.group(1)
            return None


        def merge_and_sum(intervals: np.ndarray) -> int:
            """
            intervals: Nx2 numpy array of [start, end] (inclusive, 1-based)
            returns total union length in bp.
            """
            if intervals.size == 0:
                return 0

            # sort by start then end
            intervals = intervals[np.lexsort((intervals[:,1], intervals[:,0]))]

            total = 0
            cur_start, cur_end = intervals[0]

            for s, e in intervals[1:]:
                if s <= cur_end + 1:
                    if e > cur_end:
                        cur_end = e
                else:
                    total += (cur_end - cur_start + 1)
                    cur_start, cur_end = s, e

            total += (cur_end - cur_start + 1)
            return int(total)

        # define col names of gtf file
        cols = ["seqname","source","feature","start","end","score","strand","frame","attribute"]
        gtf = pd.read_csv(
            gtf_path, sep="\t", comment="#", header=None, names=cols,
            dtype={"seqname":"category","source":"category","feature":"category",
                   "start":int,"end":int,"score":"string","strand":"category",
                   "frame":"string","attribute":"string"}
        )

        # keep only exons
        exons = gtf[gtf["feature"] == "exon"].copy()
        if exons.empty:
            return pd.DataFrame(columns=["gene_id","exonic_length"])

        # gene_id
        exons["gene_id"] = exons["attribute"].map(extract_gene_id)
        exons = exons.dropna(subset=["gene_id"])

        # for safety, ensure start <= end
        bad = (exons["end"] < exons["start"])
        if bad.any():
            exons.loc[bad, ["start","end"]] = exons.loc[bad, ["end","start"]].values

        # group by gene and merge intervals
        def _sum_for_gene(g):
            ivals = g[["start","end"]].to_numpy(dtype=np.int64)
            return merge_and_sum(ivals)

        lengths = (
            exons.groupby("gene_id", sort=False, observed=True)
                 .apply(_sum_for_gene)
                 .reset_index(name="exonic_length")
        )
        return lengths

    def calculate_fpkm(self, expr_df, lengths_df, robust=True):
        """
        expr_df: DataFrame, rows=genes, cols=samples, values=raw counts
        lengths_df: DataFrame with columns ["gene_id", "gene_length"] (bp)

        Returns: FPKM DataFrame with same shape as expr_df
        if robus = True, returns size-factor normalized fpkms, same as DESEQ2
        """

        lengths = lengths_df.set_index("gene_id")["exonic_length"]
        expr_df = expr_df.copy()

        # make sure order matches
        expr_df = expr_df.loc[expr_df.index.intersection(lengths.index)]
        lengths = lengths.loc[expr_df.index]

        if robust is False:
            library_size = expr_df.sum(axis=0)
        else:
            library_size = self.size_factors.T * np.exp(np.mean(np.log(expr_df.sum(axis=0))))

        fpkm = expr_df.div(lengths, axis=0) * 1e9
        fpkm = fpkm.div(library_size, axis=1)

        return fpkm
