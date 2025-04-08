import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from optht import optht
import copy
from pydeseq2.preprocessing import deseq2_norm

class ProtriderDataset(Dataset):
    def __init__(self, csv_file, index_col, sa_file=None, 
                 cov_used=None, log_func=np.log,
                 maxNA_filter=0.3):
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

        self.data = self.data.T
        self.data.index.names = ['sampleID']
        self.data.columns.name = 'proteinID'
        print(f'\tFinished reading raw data with shape: {self.data.shape}')
        
        # replace 0 with NaN (for proteomics intensities)
        self.data.replace(0, np.nan, inplace=True)
        
        # filter out proteins with too many NaNs
        filtered = np.mean(np.isnan(self.data), axis=0)
        self.data = (self.data.T[filtered <= maxNA_filter]).T
        self.n_samples, self.n_prot = self.data.shape
        print(f"\tFiltering out {np.sum(filtered > maxNA_filter)} proteins with too many missing values. New shape: {self.data.shape}")
        
        self.raw_data = copy.deepcopy(self.data) ## for storing output
        
        # normalize data with deseq2
        deseq_out, size_factors = deseq2_norm(self.data.replace(np.nan, 0, 
                                                                inplace=False))
        ### check that deseq2 worked, otherwise ignore
        if deseq_out.isna().sum().sum()==0:
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
        self.X = self.centered_log_data_noNA + self.prot_means ## same as data but without NAs

        ## to torch
        self.X = torch.tensor(self.X)
        #self.X_target = self.X ### needed for outlier injection
        self.mask = np.array(self.mask.values)
        self.torch_mask = torch.tensor(self.mask)
        self.prot_means_torch = torch.from_numpy( self.prot_means).squeeze(0)
        
        # sample annotation including covariates
        if sa_file is not None:
            sa_file_extension = Path(sa_file).suffix
            if sa_file_extension == '.csv':
                self.sample_anno = pd.read_csv(sa_file)
            elif sa_file_extension == '.tsv':
                self.sample_anno = pd.read_csv(sa_file, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {sa_file_extension}")    
            print(f'\tFinished reading sample annotation with shape: {self.sample_anno.shape}')
        else:
            cov_used = None
            
        if cov_used is not None:
            self.covariates = self.sample_anno.loc[:,cov_used]
            num_types = ["float64", "float32", "float16", 
                         "complex64", "complex128", "int64",
                         "int32", "int16", "int8", "uint8"]
            for col in self.covariates.columns:
                if self.covariates.loc[:,col].dtype not in num_types:
                    self.covariates[col] = pd.factorize(self.covariates[col])[0]
                    self.covariates[col] = np.where(self.covariates[col] < 0, 
                                                    np.max(self.covariates[col])+1, 
                                                    self.covariates[col])
            self.covariates = torch.tensor(self.covariates.values)
            
            # one_hot encoding of covariates
            for col in range(self.covariates.shape[1]):
                one_hot_col = F.one_hot(self.covariates[:,col], num_classes=self.covariates[:,col].max().numpy()+1) 
                try:
                    one_hot = torch.cat((one_hot, one_hot_col), dim=1)
                except:
                    one_hot = one_hot_col
            self.cov_one_hot = one_hot
        else:
            self.covariates = torch.empty(self.X.shape[0], 0)
            self.cov_one_hot = torch.empty(self.X.shape[0], 0)
        print(f'\tFinished reading covariates. No. one-hot-encoded covariates used: ', self.cov_one_hot.shape[1])
        
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.torch_mask[idx], self.cov_one_hot[idx], self.prot_means_torch)

    def _perform_svd(self):
        self.U, self.s, self.Vt = np.linalg.svd(np.hstack([self.centered_log_data_noNA,
                                                          self.cov_one_hot.detach().cpu().numpy()
                                                          ]),
                                                full_matrices=False)
        print('\tFinished fitting SVD with shapes U:', self.U.shape, 's:', self.s.shape, 'Vt:', self.Vt.shape)

    def _find_enc_dim_optht(self):
        try:
            q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None)
        except:
            self._perform_svd()
            q = optht(self.centered_log_data_noNA, sv=self.s, sigma=None)
        return q

class ProtriderSubset(Subset):
    def __getattr__(self, name):
        # fallback to original dataset's attributes
        return getattr(self.dataset, name)
