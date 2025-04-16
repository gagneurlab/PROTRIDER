import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from pandas import DataFrame
from dataclasses import dataclass
# todo add to setup.py and requirements.txt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .model import train, train_val, mse_masked, ProtriderAutoencoder
from .datasets import ProtriderDataset, ProtriderSubset, ProtriderKfoldCVGenerator, ProtriderLOOCVGenerator
from .stats import get_pvals, fit_residuals
from .model_helper import find_latent_dim, init_model

__all__ = ["ModelInfo", "Result", "run_experiment", "run_experiment_kfoldcv"]


@dataclass
class ModelInfo:
    """Stores model information."""
    q: np.array
    learning_rate: np.array
    n_epochs: np.array
    test_loss: np.array
    train_loss_history: List[np.array] = None
    val_loss_history: List[np.array] = None


@dataclass
class Result:
    """Stores results from a standard run of PROTRIDER."""
    dataset: ProtriderDataset
    df_out: pd.DataFrame
    df_res: pd.DataFrame
    df_pvals: pd.DataFrame
    df_Z: pd.DataFrame
    df_pvals_adj: pd.DataFrame
    log2fc: np.ndarray
    fc: np.ndarray
    n_out_median: int
    n_out_max: int
    n_out_total: int


def run_experiment(input_intensities, config, sample_annotation, log_func, base_fn, device) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection in a single run.
    Args:
        input_intensities:
        config:
        sample_annotation:
        log_func:
        base_fn:
        device:

    Returns:

    """

    ## 1. Initialize dataset
    print('=== Initializing dataset ===')
    dataset = ProtriderDataset(csv_file=input_intensities,
                               index_col=config['index_col'],
                               sa_file=sample_annotation,
                               cov_used=config['cov_used'],
                               log_func=log_func,
                               maxNA_filter=config['max_allowed_NAs_per_protein'],
                               device=device)

    ## 2. Find latent dim
    print('=== Finding latent dimension ===')
    q = find_latent_dim(dataset, method=config['find_q_method'],
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
                        out_dir=config['out_dir'],
                        device=device
                        )
    print(f'\tLatent dimension found with method {config["find_q_method"]}: {q}')

    ## 3. Init model with found latent dim
    model = init_model(dataset, q,
                       init_wPCA=config['init_pca'],
                       n_layer=config['n_layers'],
                       h_dim=config['h_dim'],
                       device=device
                       )
    print('\tModel:', model, 'device:', device)

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

    mu, sigma, df0 = fit_residuals(df_res.values, dis=config['pval_dist'])
    pvals, Z, pvals_adj = get_pvals(df_res.values,
                                    mu=mu,
                                    sigma=sigma,
                                    df0=df0,
                                    how=config['pval_sided'],
                                    dis=config['pval_dist'],
                                    padjust=config["pval_adj"])
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, pvals=pvals, Z=Z, pvals_adj=pvals_adj,
                             pseudocount=config['pseudocount'], outlier_threshold=config['outlier_threshold'],
                             base_fn=base_fn)
    model_info = ModelInfo(q=q, learning_rate=config['lr'], n_epochs=config['n_epochs'], test_loss=final_loss)
    return result, model_info


def run_experiment_loocv(input_intensities, config, sample_annotation, log_func, base_fn, device) -> Tuple[
    Result, ModelInfo, DataFrame]:
    """
    Perform protein outlier detection with cross-validation.
    Args:
        nfold_to_run:
        input_intensities:
        config:
        sample_annotation:
        log_func:
        base_fn:

    Returns:

    """

    ## 1. Initialize cross validation generator
    print('=== Initializing cross validation ===')
    cv_gen = ProtriderLOOCVGenerator(input_intensities, sample_annotation, config['index_col'], config['cov_used'],
                                     config['max_allowed_NAs_per_protein'], log_func, seed=config['seed'],
                                     device=device)
    dataset = cv_gen.dataset

    # test results
    pvals_list = []
    Z_list = []
    pvals_adj_list = []
    df_out_list = []
    df_res_list = []
    test_loss_list = []
    train_loss_list = []
    val_loss_list = []
    q_list = []
    folds_list = []
    ## 2. Loop over folds
    for fold, (train_subset, val_subset, test_subset) in enumerate(cv_gen):
        print(f'=== Fold {fold} ===')
        print(f'\tTrain subset size: {len(train_subset)}')
        print(f'\tValidation subset size: {len(val_subset)}')
        print(f'\tTest subset size: {len(test_subset)}')

        ## 3. Find latent dim
        print('=== Finding latent dimension ===')
        pca_subset = ProtriderSubset.concat([train_subset, val_subset])
        q = find_latent_dim(pca_subset, method=config['find_q_method'],
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
                            out_dir=config['out_dir'],
                            device=device)
        print(f'\tLatent dimension found with method {config["find_q_method"]}: {q}')

        ## 4. Init model with found latent dim
        model = init_model(pca_subset, q,
                           init_wPCA=config['init_pca'],
                           n_layer=config['n_layers'],
                           h_dim=config['h_dim'],
                           device=device)
        print('\tModel:', model, 'device:', device)

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
            train_losses, val_losses = train_val(train_subset, val_subset, model,
                                                 n_epochs=config['n_epochs'],
                                                 learning_rate=float(config['lr']),
                                                 batch_size=config['batch_size'],
                                                 patience=config['early_stopping_patience'],
                                                 min_delta=config['early_stopping_min_delta']
                                                 )
            train_loss_list.append(train_losses)
            val_loss_list.append(val_losses)
            df_out_train, train_loss = _inference(train_subset, model)
            df_out_val, val_loss = _inference(val_subset, model)
            print(f'\tFold {fold} train loss: {train_loss}')
            print(f'\tFold {fold} validation loss: {val_loss}')
            _plot_loss_history(train_losses, val_losses, fold, config['out_dir'])

        ## 7. Fit residual distribution on validation set
        print('=== Estimating residual distribution parameters on validation set ===')
        df_res_val = val_subset.data - df_out_val  # log data - pred data
        mu, sigma, df0 = fit_residuals(df_res_val.values, dis=config['pval_dist'])

        # 8. Compute pvals on test set
        print('=== Running model on test set ===')
        df_out_test, loss = _inference(test_subset, model)
        print(f'\tFold {fold} test loss: {train_loss}')
        df_res_test = test_subset.data - df_out_test  # log data - pred data
        pvals, Z, pvals_adj = get_pvals(df_res_test.values, mu=mu, sigma=sigma, df0=df0,
                                        how=config['pval_sided'], padjust=config["pval_adj"])
        df_out_list.append(df_out_test)
        df_res_list.append(df_res_test)
        test_loss_list.append(loss)
        q_list.append(q)
        pvals_list.append(pvals)
        Z_list.append(Z)
        pvals_adj_list.append(pvals_adj)
        folds_list.extend([fold] * len(pvals))

    pvals = np.concatenate(pvals_list)
    Z = np.concatenate(Z_list)
    pvals_adj = np.concatenate(pvals_adj_list)
    df_out = pd.concat(df_out_list)
    df_res = pd.concat(df_res_list)
    df_folds = pd.DataFrame({'fold': folds_list, }, index=df_out.index)

    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, pvals=pvals,
                             Z=Z, pvals_adj=pvals_adj, pseudocount=config['pseudocount'],
                             outlier_threshold=config['outlier_threshold'], base_fn=base_fn)
    model_info = ModelInfo(q=np.array(q_list), learning_rate=np.array(config['lr']),
                           n_epochs=np.array(config['n_epochs']), test_loss=np.array(test_loss_list),
                           train_loss_history=train_loss_list, val_loss_history=val_loss_list)
    return result, model_info, df_folds


def run_experiment_kfoldcv(input_intensities, config, sample_annotation, log_func, base_fn, device) -> Tuple[
    Result, ModelInfo, DataFrame]:
    """
    Perform protein outlier detection with cross-validation.
    Args:
        nfold_to_run:
        input_intensities:
        config:
        sample_annotation:
        log_func:
        base_fn:

    Returns:

    """

    ## 1. Initialize cross validation generator
    print('=== Initializing cross validation ===')
    cv_gen = ProtriderKfoldCVGenerator(input_intensities, sample_annotation, config['index_col'], config['cov_used'],
                                       config['max_allowed_NAs_per_protein'], log_func, num_folds=config['n_folds'],
                                       seed=config['seed'], device=device)
    dataset = cv_gen.dataset

    # test results
    pvals_list = []
    Z_list = []
    pvals_adj_list = []
    df_out_list = []
    df_res_list = []
    test_loss_list = []
    train_loss_list = []
    val_loss_list = []
    q_list = []
    folds_list = []
    ## 2. Loop over folds
    for fold, (pca_subset, train_subset, val_subset, test_subset) in enumerate(cv_gen):
        print(f'=== Fold {fold} ===')
        print(f'\tPCA subset size: {len(pca_subset)}')
        print(f'\tTrain subset size: {len(train_subset)}')
        print(f'\tValidation subset size: {len(val_subset)}')
        print(f'\tTest subset size: {len(test_subset)}')

        ## 3. Find latent dim
        print('=== Finding latent dimension ===')
        q = find_latent_dim(pca_subset, method=config['find_q_method'],
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
                            out_dir=config['out_dir'],
                            device=device)
        print(f'\tLatent dimension found with method {config["find_q_method"]}: {q}')

        ## 4. Init model with found latent dim
        model = init_model(pca_subset, q,
                           init_wPCA=config['init_pca'],
                           n_layer=config['n_layers'],
                           h_dim=config['h_dim'],
                           device=device)
        print('\tModel:', model, 'device:', device)

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
            train_losses, val_losses = train_val(train_subset, val_subset, model,
                                                 n_epochs=config['n_epochs'],
                                                 learning_rate=float(config['lr']),
                                                 batch_size=config['batch_size'],
                                                 patience=config['early_stopping_patience'],
                                                 min_delta=config['early_stopping_min_delta']
                                                 )
            train_loss_list.append(train_losses)
            val_loss_list.append(val_losses)
            df_out_train, train_loss = _inference(train_subset, model)
            df_out_val, val_loss = _inference(val_subset, model)
            print(f'\tFold {fold} train loss: {train_loss}')
            print(f'\tFold {fold} validation loss: {val_loss}')
            _plot_loss_history(train_losses, val_losses, fold, config['out_dir'])

        ## 7. Fit residual distribution on validation set
        print('=== Estimating residual distribution parameters on validation set ===')
        df_res_val = val_subset.data - df_out_val  # log data - pred data
        mu, sigma, df0 = fit_residuals(df_res_val.values, dis=config['pval_dist'])

        # 8. Compute pvals on test set
        print('=== Running model on test set ===')
        df_out_test, loss = _inference(test_subset, model)
        print(f'\tFold {fold} test loss: {train_loss}')
        df_res_test = test_subset.data - df_out_test  # log data - pred data
        pvals, Z, pvals_adj = get_pvals(df_res_test.values, mu=mu, sigma=sigma, df0=df0,
                                        how=config['pval_sided'], padjust=config["pval_adj"])
        df_out_list.append(df_out_test)
        df_res_list.append(df_res_test)
        test_loss_list.append(loss)
        q_list.append(q)
        pvals_list.append(pvals)
        Z_list.append(Z)
        pvals_adj_list.append(pvals_adj)
        folds_list.extend([fold] * len(pvals))

    pvals = np.concatenate(pvals_list)
    Z = np.concatenate(Z_list)
    pvals_adj = np.concatenate(pvals_adj_list)
    df_out = pd.concat(df_out_list)
    df_res = pd.concat(df_res_list)
    df_folds = pd.DataFrame({'fold': folds_list, }, index=df_out.index)

    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, pvals=pvals,
                             Z=Z, pvals_adj=pvals_adj, pseudocount=config['pseudocount'],
                             outlier_threshold=config['outlier_threshold'], base_fn=base_fn)
    model_info = ModelInfo(q=np.array(q_list), learning_rate=np.array(config['lr']),
                           n_epochs=np.array(config['n_epochs']), test_loss=np.array(test_loss_list),
                           train_loss_history=train_loss_list, val_loss_history=val_loss_list)
    return result, model_info, df_folds


def _plot_loss_history(train_losses, val_losses, fold, out_dir):
    # plot the loss history; stratified by fold
    plot_dir = Path(out_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_p = f'{plot_dir}/loss_history_fold{fold}.png'

    df = pd.concat([pd.DataFrame(dict(type='validation', loss=val_losses, epoch=np.arange(len(val_losses)))),
                    pd.DataFrame(dict(type='train', loss=train_losses, epoch=np.arange(len(train_losses))))])
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='loss', hue='type')
    plt.title(f'Loss history for fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title=f'Fold {fold}')
    plt.savefig(out_p)
    plt.close()
    print(f"\t Saved loss history plot for fold {fold} to {out_p}")


def _inference(dataset: Union[ProtriderDataset, ProtriderSubset], model: ProtriderAutoencoder):
    X_out = model(dataset.X,
                  prot_means=dataset.prot_means_torch, cond=dataset.cov_one_hot)
    loss = mse_masked(dataset.X, X_out, dataset.torch_mask).detach().cpu().numpy()

    df_out = pd.DataFrame(X_out.detach().cpu().numpy())
    df_out.columns = dataset.data.columns
    df_out.index = dataset.data.index

    return df_out, loss


def _format_results(df_out, df_res, pvals, Z, pvals_adj, dataset, pseudocount, outlier_threshold, base_fn):
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
    n_out_median = np.nanmedian(outs_per_sample)
    n_out_max = np.nanmax(outs_per_sample)
    n_out_total = np.nansum(outs_per_sample)
    print(f'\tFinished computing pvalues. No. outliers per sample in median: {n_out_median}')
    print(f'\t {sorted(outs_per_sample)}')

    print(f'\tFinished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')
    print(f'\t {sorted(outs_per_sample)}')

    return Result(dataset=dataset, df_out=df_out, df_res=df_res, df_pvals=df_pvals, df_Z=df_Z,
                  df_pvals_adj=df_pvals_adj, log2fc=log2fc, fc=fc, n_out_median=n_out_median, n_out_max=n_out_max,
                  n_out_total=n_out_total)
