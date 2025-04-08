import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path
import pprint
import click
from typing import Callable, Iterable, Union
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split

from .model import train, mse_masked, ProtriderAutoencoder
from .datasets import ProtriderDataset, ProtriderSubset
from .stats import get_pvals, fit_residuals, get_pvals_cv
from .model_helper import _find_latent_dim, _init_model


class ProtriderCVGenerator:
    """
    Cross-validation generator for the ProtriderDataset.
    Creates train, validation, and test splits for k-fold cross validation.
    """

    def __init__(self, input_intensities: str, sample_annotation: str, index_col: str,
                 cov_used: Iterable[str], maxNA_filter: float,
                 log_func: Callable[[ArrayLike], ArrayLike], num_folds: int = 5, seed: int = 42):
        """
        Args:
            input_intensities: Path to CSV file with protein intensity data
            sample_annotation: Path to CSV file with sample annotations
            index_col: Name of the index column
            cov_used: List of covariates to use
            maxNA_filter: Maximum proportion of NAs allowed per protein
            log_func: Log function to apply to the data
            num_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
        """
        self.input_intensities = input_intensities
        self.sample_annotation = sample_annotation
        self.index_col = index_col
        self.cov_used = cov_used
        self.maxNA_filter = maxNA_filter
        self.log_func = log_func
        self.num_folds = num_folds
        self.seed = seed

        # Initialize the dataset
        print('=== Initializing dataset ===')
        self.dataset = ProtriderDataset(csv_file=input_intensities,
                                        index_col=index_col,
                                        sa_file=sample_annotation,
                                        cov_used=cov_used,
                                        log_func=log_func,
                                        maxNA_filter=maxNA_filter)

        # Set up KFold
        self.kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

        # Pre-compute all folds for consistency
        self._folds = list(self.kf.split(self.dataset))

    def __iter__(self):
        """Generate train, validation, and test subsets for each fold"""
        for train_val_idx, test_idx in self._folds:
            # Split training data into train and validation
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25,
                                                  random_state=self.seed)

            # Create subsets
            train_subset = ProtriderSubset(self.dataset, train_idx)
            val_subset = ProtriderSubset(self.dataset, val_idx)
            test_subset = ProtriderSubset(self.dataset, test_idx)

            yield train_subset, val_subset, test_subset

    def __len__(self):
        """Return the number of folds"""
        return self.num_folds


# @click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.command()
@click.option(
    "--config",
    default='../config.yaml',
    help="Configuration file containing PROTRIDER custom options",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--input_intensities",
    help="csv input file containing intensities. Columns are samples and rows are proteins. See example here: FIXME",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--sample_annotation",
    help="csv file containing sample annotations",
    type=click.Path(exists=True, dir_okay=False),
)
def main(config, input_intensities: str, sample_annotation: str = None) -> None:
    """# PROTRIDER

    PROTRIDER is a package for calling protein outliers on mass spectrometry data

    Links:
    - Publication: FIXME
    - Official code repository: https://github.com/gagneurlab/PROTRIDER

    """

    ## Load config with params
    config = yaml.load(open(config), Loader=yaml.FullLoader)
    print('++++ STARTING PROTRIDER ++++\n Config used: ')
    pprint.pprint(config)

    if config['out_dir'] is not None:
        path = Path(config['out_dir'])
        path.mkdir(parents=True, exist_ok=True)

    if config['log_func_name'] == "log2":
        log_func = np.log2
        base_fn = lambda x: 2 ** x
    elif config['log_func_name'] == "log10":
        log_func = np.log10
        base_fn = lambda x: 10 ** x
    elif config['log_func_name'] == 'log':
        log_func = np.log
        base_fn = np.exp
    else:
        if config['log_func_name'] is None:
            print('[WARNING] No log function passed for preprocessing. \nAssuming data is already log transformed.')
            log_func = lambda x: x  # id
            base_fn = np.exp
        else:
            raise ValueError(f"Log func {config['log_func_name']} not supported.")

    ## Catch some errors/inconsistencies
    if config['find_q_method'] == 'OHT' and config['cov_used'] is not None:
        raise ValueError('OHT not implemented with covariate inclusion yet')

    if config['cross_val']:
        if config['find_q_method'] != 'OHT':
            raise ValueError('Cross-validation only implemented with OHT method')
        run_fun = _run_cv
    else:
        run_fun = _run

    dataset, df_out, df_res, q, final_loss, pvals, Z, pvals_adj = run_fun(input_intensities, config, sample_annotation,
                                                                          log_func)

    df_pvals, df_Z, df_pvals_adj, log2fc, fc = _format_results(df_out=df_out, pvals=pvals, Z=Z,
                                                               pvals_adj=pvals_adj, dataset=dataset,
                                                               pseudocount=config['pseudocount'],
                                                               outlier_threshold=config['outlier_threshold'],
                                                               base_fn=base_fn)
    if config['out_dir'] is not None:
        _write_results(dataset=dataset, q=q, final_loss=final_loss, df_out=df_out, df_res=df_res, df_pvals=df_pvals,
                       df_Z=df_Z, df_pvals_adj=df_pvals_adj, fc=fc, log2fc=log2fc, out_dir=config['out_dir'])

    return _report_summary(dataset.raw_data, dataset.data, df_out, df_Z,
                           df_pvals, df_pvals_adj, log2fc, fc,
                           config['pval_dist'], config['outlier_threshold'],
                           config['out_dir'], config['report_all'])


def _inference(dataset: Union[ProtriderDataset, ProtriderSubset], model: ProtriderAutoencoder):
    X_out = model(dataset.X,
                  prot_means=dataset.prot_means_torch, cond=dataset.cov_one_hot)
    loss = mse_masked(dataset.X, X_out, dataset.torch_mask).detach().numpy()

    df_out = pd.DataFrame(X_out.detach().numpy())
    df_out.columns = dataset.data.columns
    df_out.index = dataset.data.index

    return df_out, loss


def _run(input_intensities, config, sample_annotation, log_func):
    ## 1. Initialize dataset
    print('=== Initializing dataset ===')
    dataset = ProtriderDataset(csv_file=input_intensities,
                               index_col=config['index_col'],
                               sa_file=sample_annotation,
                               cov_used=config['cov_used'],
                               log_func=log_func,
                               maxNA_filter=config['max_allowed_NAs_per_protein'])

    ## 2. Find latent dim
    print('=== Finding latent dimension ===')
    q = _find_latent_dim(dataset, method=config['find_q_method'],
                         ### Params for grid search method
                         inj_freq=float(config['inj_freq']),
                         inj_mean=config['inj_mean'],
                         inj_sd=config['inj_sd'],
                         seed=config['seed'],
                         init_wPCA=config['init_pca'],
                         n_layers=config['n_layers'],
                         h_dim=config['h_dim'],
                         n_epochs=config['n_epochs'],
                         learning_rate=float(config['lr']),
                         batch_size=config['batch_size'],
                         pval_sided=config['pval_sided'],
                         pval_dist=config['pval_dist'],
                         out_dir=config['out_dir']
                         )
    print(f'\tLatent dimension found with method {config["find_q_method"]}: {q}')

    ## 3. Init model with found latent dim
    model = _init_model(dataset, q,
                        init_wPCA=config['init_pca'],
                        n_layer=config['n_layers'],
                        h_dim=config['h_dim']
                        )
    print('\tModel:', model)

    ## 4. Compute initial MSE loss
    df_out, final_loss = _inference(dataset, model)
    print('\tInitial loss after model init: ', final_loss)
    if config['autoencoder_training']:
        print('=== Fitting model ===')
        ## 5. Train model
        train(dataset, model,
              n_epochs=config['n_epochs'],
              learning_rate=float(config['lr']),
              batch_size=config['batch_size'],
              )  # .detach().numpy()
        df_out, final_loss = _inference(dataset, model)
        print('Final loss:', final_loss)

    ## 6. Compute residuals, pvals, zscores
    print('=== Computing statistics ===')
    df_res = dataset.data - df_out  # log data - pred data
    pvals, Z, pvals_adj = get_pvals(df_res.values,
                                    how=config['pval_sided'],
                                    dis=config['pval_dist'],
                                    padjust=config["pval_adj"])

    return dataset, df_out, df_res, q, final_loss, pvals, Z, pvals_adj


def _run_cv(input_intensities, config, sample_annotation, log_func):
    ## 1. Initialize cross validation generator
    print('=== Initializing cross validation generator ===')
    cv_gen = ProtriderCVGenerator(input_intensities, sample_annotation,
                                  config['index_col'], config['cov_used'],
                                  config['max_allowed_NAs_per_protein'],
                                  log_func)
    dataset = cv_gen.dataset

    # test results
    pvals_list = []
    Z_list = []
    pvals_adj_list = []
    df_out_list = []
    df_res_list = []
    loss_list = []
    ## 2. Loop over folds
    for fold, (train_subset, val_subset, test_subset) in enumerate(cv_gen):
        print(f'=== Fold {fold + 1} ===')
        print(
            f'Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}, Test subset size: {len(test_subset)}')

        ## 3. Find latent dim
        print('=== Finding latent dimension ===')
        q = _find_latent_dim(train_subset, method=config['find_q_method'],
                             ### Params for grid search method
                             inj_freq=float(config['inj_freq']),
                             inj_mean=config['inj_mean'],
                             inj_sd=config['inj_sd'],
                             seed=config['seed'],
                             init_wPCA=config['init_pca'],
                             n_layers=config['n_layers'],
                             h_dim=config['h_dim'],
                             n_epochs=config['n_epochs'],
                             learning_rate=float(config['lr']),
                             batch_size=config['batch_size'],
                             pval_sided=config['pval_sided'],
                             pval_dist=config['pval_dist'],
                             out_dir=config['out_dir']
                             )
        print(f'\tLatent dimension found with method {config["find_q_method"]}: {q}')

        ## 4. Init model with found latent dim
        model = _init_model(train_subset, q,
                            init_wPCA=config['init_pca'],
                            n_layer=config['n_layers'],
                            h_dim=config['h_dim']
                            )
        print('\tModel:', model)

        ## 5. Compute initial MSE loss
        df_out_train, train_loss = _inference(train_subset, model)
        df_out_val, val_loss = _inference(val_subset, model)
        print(f'\tTrain loss after model init: {train_loss}')
        print(f'\tValidation loss after model init: {val_loss}')

        if config['autoencoder_training']:
            print('=== Fitting model ===')
            ## 6. Train model
            # todo train validate (hyperparameter tuning)
            # todo pass validation set as well
            train(train_subset, model,
                  n_epochs=config['n_epochs'],
                  learning_rate=float(config['lr']),
                  batch_size=config['batch_size'],
                  )
            df_out_train, train_loss = _inference(train_subset, model)
            df_out_val, val_loss = _inference(val_subset, model)
            print(f'\tFinal train loss: {train_loss}')
            print(f'\tFinal validation loss: {val_loss}')

        ## 7. Fit residual distribution on validation set
        print('=== Estimating residual distribution parameters on validation set ===')
        df_res_val = val_subset.data - df_out_val  # log data - pred data
        dist_params = fit_residuals(df_res_val.values, dis=config['pval_dist'])

        # 8. Compute pvals on test set
        print('=== Running model on test set ===')
        df_out_test, loss = _inference(test_subset, model)
        df_res_test = test_subset.data - df_out_test  # log data - pred data
        pvals, Z, pvals_adj = get_pvals_cv(df_res_test.values, how=config['pval_sided'], padjust=config["pval_adj"],
                                           dist_params=dist_params)
        df_out_list.append(df_out_test)
        df_res_list.append(df_res_test)
        loss_list.append(loss)
        pvals_list.append(pvals)
        Z_list.append(Z)
        pvals_adj_list.append(pvals_adj)

    pvals = np.concatenate(pvals_list)
    Z = np.concatenate(Z_list)
    pvals_adj = np.concatenate(pvals_adj_list)
    losses = np.concatenate(loss_list)
    df_out = pd.concat(df_out_list)
    df_res = pd.concat(df_res_list)

    # todo fix this
    final_loss = np.mean(losses)

    return dataset, df_out, df_res, q, final_loss, pvals, Z, pvals_adj


def _format_results(df_out, pvals, Z, pvals_adj, dataset, pseudocount, outlier_threshold, base_fn):
    # Store as df
    df_pvals_adj = pd.DataFrame(pvals_adj)
    df_pvals_adj.columns = dataset.data.columns
    df_pvals_adj.index = dataset.data.index

    # Store as df
    df_pvals = pd.DataFrame(pvals)
    df_pvals.columns = dataset.data.columns
    df_pvals.index = dataset.data.index

    df_Z = pd.DataFrame(Z)
    df_Z.columns = dataset.data.columns
    df_Z.index = dataset.data.index

    pseudocount = pseudocount  # 0.01
    log2fc = np.log2(base_fn(dataset.data) + pseudocount) - np.log2(base_fn(df_out) + pseudocount)
    fc = (base_fn(dataset.data) + pseudocount) / (base_fn(df_out) + pseudocount)

    outs_per_sample = np.sum(df_pvals_adj.values <= outlier_threshold, axis=1)

    print(f'\tFinished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')
    print(f'\t {sorted(outs_per_sample)}')

    return df_pvals, df_Z, df_pvals_adj, log2fc, fc


def _write_results(dataset, q, final_loss, df_out, df_res, df_pvals, df_Z, df_pvals_adj, fc, log2fc, out_dir):
    if out_dir is not None:
        print('=== Saving output ===')
        out_dir = out_dir

        # AE input
        out_p = f'{out_dir}/processed_input.csv'
        dataset.data.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved processed_input to {out_p}")

        # AE output
        out_p = f'{out_dir}/output.csv'
        df_out.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved output to {out_p}")

        # residuals
        out_p = f'{out_dir}/residuals.csv'
        df_res.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved residuals to {out_p}")

        # p-values
        out_p = f'{out_dir}/pvals.csv'
        df_pvals.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved P-values to {out_p}")

        # p-values adj
        out_p = f'{out_dir}/pvals_adj.csv'
        df_pvals_adj.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved adjusted P-values to {out_p}")

        # Z-scores
        out_p = f'{out_dir}/zscores.csv'
        df_Z.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved z scores to {out_p}")

        # log2fc
        out_p = f'{out_dir}/log2fc.csv'
        log2fc.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved log2fc scores to {out_p}")

        # fc
        out_p = f'{out_dir}/fc.csv'
        fc.T.to_csv(out_p, header=True, index=True)
        print(f"\t Saved fc scores to {out_p}")

        # latent space
        # FIXME

        # Additional info
        out_p = f'{out_dir}/additional_info.csv'
        df_info = pd.DataFrame([[q, final_loss]], columns=["opt_q", "final_loss"])
        df_info.to_csv(out_p, header=True, index=True)
        print(f"\t Saved additional input to {out_p}")


def _report_summary(raw_in, ae_in, ae_out, zscores,
                    pvals, pvals_adj,
                    log2fc, fc,
                    pval_dist='gaussian',
                    outlier_thres=0.1, out_dir=None, include_all=False):
    print('=== Reporting summary ===')
    ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
              .rename(columns={'value': 'PROTEIN_EXPECTED_LOG2INT'}))
    ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
             .rename(columns={'value': 'PROTEIN_LOG2INT'}))
    raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
              .rename(columns={'value': 'PROTEIN_INT'}))
    zscores = (zscores.reset_index().melt(id_vars='sampleID')
               .rename(columns={'value': 'PROTEIN_ZSCORE'}))
    pvals = (pvals.reset_index().melt(id_vars='sampleID')
             .rename(columns={'value': 'PROTEIN_PVALUE'}))
    pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'value': 'PROTEIN_PADJ'}))
    log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
              .rename(columns={'value': 'PROTEIN_LOG2FC'}))
    fc = (fc.reset_index().melt(id_vars='sampleID')
          .rename(columns={'value': 'PROTEIN_FC'}))

    merge_cols = ['sampleID', 'proteinID']
    df_res = (ae_in.merge(ae_out, on=merge_cols)
              .merge(raw_in, on=merge_cols)
              .merge(zscores, on=merge_cols)
              .merge(pvals, on=merge_cols)
              .merge(pvals_adj, on=merge_cols)
              .merge(log2fc, on=merge_cols)
              .merge(fc, on=merge_cols)
              ).reset_index(drop=True)

    df_res['PROTEIN_outlier'] = df_res['PROTEIN_PADJ'].apply(lambda x: x <= outlier_thres)
    df_res['pvalDistribution'] = pval_dist

    if not include_all:
        original_len = df_res.shape[0]
        df_res = df_res.query('PROTEIN_outlier==True')
        print(
            f'\t--- Removing non-significant sample-protein combinations. \n\tOriginal len: {original_len}, new len: {df_res.shape[0]}---')

    if out_dir is not None:
        out_p = f'{out_dir}/protrider_summary.csv'
        df_res.to_csv(out_p, index=None)
        print(f'\t--- Wrote output summary with shape {df_res.shape} to <{out_p}>---')
    return df_res


if __name__ == '__main__':
    main()
