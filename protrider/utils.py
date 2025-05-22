import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from pandas import DataFrame
from dataclasses import dataclass
from pathlib import Path
import logging

import torch

from .model import train, train_val, MSEBCELoss, ProtriderAutoencoder
from .datasets import ProtriderDataset, ProtriderSubset, ProtriderKfoldCVGenerator, ProtriderLOOCVGenerator
from .stats import get_pvals, fit_residuals, adjust_pvals
from .model_helper import find_latent_dim, init_model

__all__ = ["ModelInfo", "Result", "run_experiment", "run_experiment_cv"]

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Stores model information."""
    q: np.array
    learning_rate: np.array
    n_epochs: np.array
    test_loss: np.array
    train_losses: np.array


@dataclass
class Result:
    """Stores results from a standard run of PROTRIDER."""
    dataset: ProtriderDataset
    df_out: pd.DataFrame
    df_res: pd.DataFrame
    df_pvals: pd.DataFrame
    df_pvals_one_sided: pd.DataFrame 
    df_presence: pd.DataFrame
    df_Z: pd.DataFrame
    df_pvals_adj: pd.DataFrame
    log2fc: np.ndarray
    fc: np.ndarray
    n_out_median: int
    n_out_max: int
    n_out_total: int


def run_experiment(config, log_func, base_fn, device) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection in a single run.
    Args:
        config:
        log_func:
        base_fn:
        device:

    Returns:

    """
    ## 1. Initialize dataset
    logger.info('Initializing dataset')
    dataset = ProtriderDataset(csv_file=config['input_intensities'],
                               index_col=config['index_col'],
                               sa_file=config['sample_annotation'],
                               cov_used=config['cov_used'],
                               log_func=log_func,
                               maxNA_filter=config['max_allowed_NAs_per_protein'],
                               device=device)

    ## 2. Find latent dim
    logger.info('Finding latent dimension')
    q = find_latent_dim(dataset, method=config['find_q_method'],
                        ### Params for grid search method
                        inj_freq=config['inj_freq'],
                        inj_mean=config['inj_mean'],
                        inj_sd=config['inj_sd'],
                        init_wPCA=config['init_pca'],
                        n_layers=config['n_layers'],
                        h_dim=config['h_dim'],
                        n_epochs=config.get('gs_epochs', config.get('n_epochs', None)),
                        learning_rate=config['lr'],
                        batch_size=config['batch_size'],
                        pval_sided=config['pval_sided'],
                        pval_dist=config['pval_dist'],
                        out_dir=config['out_dir'],
                        device=device,
                        presence_absence=config['presence_absence'],
                        lambda_bce=config['lambda_presence_absence']
                        )

    logger.info(f'Latent dimension found with method {config["find_q_method"]}: {q}')

    ## 3. Init model with found latent dim
    model = init_model(dataset, q,
                       init_wPCA=config['init_pca'],
                       n_layer=config['n_layers'],
                       h_dim=config['h_dim'],
                       device=device,
                       presence_absence=config['presence_absence'] if config['n_layers'] == 1 else False
                       )
    criterion = MSEBCELoss(presence_absence=config['presence_absence'], lambda_bce=config['lambda_presence_absence'])
    logger.info('Model:\n%s', model)
    logger.info('Device: %s', device)

    ## 4. Compute initial loss
    df_out, df_presence, init_loss, init_mse_loss, init_bce_loss = _inference(dataset, model, criterion)
    logger.info('Initial loss after model init: %s, mse loss: %s, bce loss: %s', init_loss, init_mse_loss,
                init_bce_loss)
    final_loss = 10**4
    train_losses = []
    if config['autoencoder_training']:
        logger.info('Fitting model')
        ## 5. Train model
        running_loss, running_mse_loss, running_bce_loss, train_losses = train(dataset, model, criterion, n_epochs=config['n_epochs'], learning_rate=float(config['lr']),
              batch_size=config['batch_size'])
        df_out, df_presence, final_loss, final_mse_loss, final_bce_loss = _inference(dataset, model, criterion)
        logger.info('Final loss: %s, mse loss: %s, bce loss: %s', final_loss, final_mse_loss, final_bce_loss)

    ## 6. Compute residuals, pvals, zscores
    logger.info('Computing statistics')
    df_res = dataset.data - df_out  # log data - pred data

    mu, sigma, df0 = fit_residuals(df_res.values, dis=config['pval_dist'])
    pvals, Z = get_pvals(df_res.values,
                         mu=mu,
                         sigma=sigma,
                         df0=df0,
                         how=config['pval_sided'],
                         dis=config['pval_dist'])
    pvals_one_sided, _ = get_pvals(df_res.values,
                         mu=mu,
                         sigma=sigma,
                         df0=df0,
                         how='left',
                         dis=config['pval_dist'])

    pvals_adj = adjust_pvals(pvals, method=config["pval_adj"])
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config['pseudocount'], outlier_threshold=config['outlier_threshold'],
                             base_fn=base_fn)
    model_info = ModelInfo(q=np.array(q), learning_rate=np.array(config['lr']),
                           n_epochs=np.array(config['n_epochs']), test_loss=np.array(final_loss), train_losses=train_losses)
    return result, model_info


def run_experiment_cv(config, log_func, base_fn, device) -> Tuple[
    Result, ModelInfo, DataFrame]:
    """
    Perform protein outlier detection with cross-validation.
    Args:
        config:
        log_func:
        base_fn:
        device:

    Returns:

    """
    # If fit_every_fold is set to True, the model will estimate the residual distribution parameters on every fold
    # using the train-val set.
    # If set to False, the model will estimate the residual distribution parameters on the final test residuals
    fit_every_fold = config.get('fit_every_fold', True)

    ## 1. Initialize cross validation generator
    logger.info('Initializing cross validation')
    if config.get('n_folds', None) is not None:
        cv_gen = ProtriderKfoldCVGenerator(config['input_intensities'], config['sample_annotation'], config['index_col'],
                                           config['cov_used'], config['max_allowed_NAs_per_protein'], log_func,
                                           num_folds=config['n_folds'], device=device)
    else:
        cv_gen = ProtriderLOOCVGenerator(config['input_intensities'], config['sample_annotation'], config['index_col'], config['cov_used'],
                                         config['max_allowed_NAs_per_protein'], log_func, device=device)
    dataset = cv_gen.dataset
    criterion = MSEBCELoss(presence_absence=config['presence_absence'], lambda_bce=config['lambda_presence_absence'])
    # test results
    pvals_list = []
    Z_list = []
    df_out_list = []
    df_res_list = []
    df_presence_list = []
    test_loss_list = []
    q_list = []
    folds_list = []
    ## 2. Loop over folds
    for fold, (train_subset, val_subset, test_subset) in enumerate(cv_gen):

        logger.info(f'Fold {fold}')
        logger.info(f'Train subset size: {len(train_subset)}')
        logger.info(f'Validation subset size: {len(val_subset)}')
        logger.info(f'Test subset size: {len(test_subset)}')

        ## 3. Find latent dim
        logger.info('Finding latent dimension')
        pca_subset = ProtriderSubset.concat([train_subset, val_subset])
        q = find_latent_dim(pca_subset, method=config['find_q_method'],
                            ### Params for grid search method
                            inj_freq=config['inj_freq'],
                            inj_mean=config['inj_mean'],
                            inj_sd=config['inj_sd'],
                            init_wPCA=config['init_pca'],
                            n_layers=config['n_layers'],
                            h_dim=config['h_dim'],
                            n_epochs=config.get('gs_epochs', config.get('n_epochs', None)),
                            learning_rate=config['lr'],
                            batch_size=config['batch_size'],
                            pval_sided=config['pval_sided'],
                            pval_dist=config['pval_dist'],
                            out_dir=config['out_dir'],
                            device=device,
                            presence_absence=config['presence_absence'],
                            lambda_bce=config['lambda_presence_absence']
                            )
        logger.info(f'Latent dimension found with method {config["find_q_method"]}: {q}')

        ## 4. Init model with found latent dim
        model = init_model(train_subset, q, init_wPCA=config['init_pca'], n_layer=config['n_layers'],
                           h_dim=config['h_dim'], device=device, presence_absence=config['presence_absence'])

        logger.info('Model:\n%s', model)
        logger.info('Device: %s', device)

        ## 5. Compute initial MSE loss
        df_out_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(val_subset, model, criterion)
        logger.info(f'Train loss after model init: {train_loss}')
        logger.info(f'Validation loss after model init: {val_loss}')
        if config['autoencoder_training']:
            logger.info('Fitting model')
            ## 6. Train model
            # todo train validate (hyperparameter tuning)
            # todo pass validation set as well
            train_losses, val_losses = train_val(train_subset, val_subset, model, criterion,
                                                 n_epochs=config['n_epochs'],
                                                 learning_rate=float(config['lr']),
                                                 batch_size=config['batch_size'],
                                                 patience=config.get('early_stopping_patience', 50),
                                                 min_delta=config.get('early_stopping_min_delta', 0.0001))
            _plot_loss_history(train_losses, val_losses, fold, config['out_dir'])

        df_out_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(val_subset, model, criterion)
        logger.info(f'Fold {fold} train loss: {train_loss}')
        logger.info(f'Fold {fold} validation loss: {val_loss}')

        # 7. Compute residuals on test set
        logger.info('Running model on test set')
        df_out_test, df_presence_test, test_loss, test_mse_loss, test_bce_loss = _inference(test_subset, model,
                                                                                            criterion)
        logger.info(f'Fold {fold} test loss: {test_loss}')
        df_res_test = test_subset.data - df_out_test  # log data - pred data

        df_out_list.append(df_out_test)
        if df_presence_test is not None:
            df_presence_list.append(df_presence_test)
        df_res_list.append(df_res_test)
        test_loss_list.append(test_loss)
        q_list.append(q)
        folds_list.extend([fold] * len(df_out_test))

        if fit_every_fold:
            # 8. Fit residual distribution on train-val set
            logger.info('Estimating residual distribution parameters on train-val set')
            df_res_val = val_subset.data - df_out_val  # log data - pred data
            df_res_train = train_subset.data - df_out_train  # log data - pred data
            mu, sigma, df0 = fit_residuals(pd.concat([df_res_train, df_res_val]).values, dis=config['pval_dist'])
            pvals, Z = get_pvals(df_res_test.values, mu=mu, sigma=sigma, df0=df0, how=config['pval_sided'])
            pvals_list.append(pvals)
            Z_list.append(Z)

    df_res = pd.concat(df_res_list)
    df_out = pd.concat(df_out_list)
    df_presence = pd.concat(df_presence_list) if df_presence_list != [] else None
    df_folds = pd.DataFrame({'fold': folds_list, }, index=df_out.index)

    # Compute p-values on test set
    if fit_every_fold:
        pvals = np.concatenate(pvals_list)
        Z = np.concatenate(Z_list)
    else:
        logger.info('Estimating residual distribution parameters')
        mu, sigma, df0 = fit_residuals(df_res.values, dis=config['pval_dist'])
        pvals, Z = get_pvals(df_res.values, mu=mu, sigma=sigma, df0=df0,
                             how=config['pval_sided'])

    pvals_adj = adjust_pvals(pvals, method=config["pval_adj"])
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_adj=pvals_adj, pseudocount=config['pseudocount'],
                             outlier_threshold=config['outlier_threshold'], base_fn=base_fn)
    model_info = ModelInfo(q=np.array(q_list), learning_rate=np.array(config['lr']),
                           n_epochs=np.array(config['n_epochs']), test_loss=np.array(test_loss_list))
    return result, model_info, df_folds


def _plot_loss_history(train_losses, val_losses, fold, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    logger.info(f"Saved loss history plot for fold {fold} to {out_p}")


def _inference(dataset: Union[ProtriderDataset, ProtriderSubset], model: ProtriderAutoencoder, criterion: MSEBCELoss):
    X_out = model(dataset.X, dataset.torch_mask, cond=dataset.cov_one_hot)

    loss, mse_loss, bce_loss = criterion(X_out, dataset.X, dataset.torch_mask, detached=True)
    df_presence = None
    if model.presence_absence:
        presence_hat = torch.sigmoid(X_out[1])  # Predicted presence (0–1)
        X_out = X_out[0]  # Predicted intensities

        df_presence = pd.DataFrame(presence_hat.detach().cpu().numpy())
        df_presence.columns = dataset.data.columns
        df_presence.index = dataset.data.index

    df_out = pd.DataFrame(X_out.detach().cpu().numpy())
    df_out.columns = dataset.data.columns
    df_out.index = dataset.data.index

    return df_out, df_presence, loss, mse_loss, bce_loss


def _format_results(df_out, df_res, df_presence, pvals, Z, pvals_one_sided, pvals_adj, dataset, pseudocount, outlier_threshold, base_fn):
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
    
    df_pvals_one_sided = pd.DataFrame(pvals_one_sided)
    df_pvals_one_sided.columns = dataset.data.columns
    df_pvals_one_sided.index = dataset.data.index

    pseudocount = pseudocount  # 0.01
    log2fc = np.log2(base_fn(dataset.data) + pseudocount) - np.log2(base_fn(df_out) + pseudocount)
    fc = (base_fn(dataset.data) + pseudocount) / (base_fn(df_out) + pseudocount)

    outs_per_sample = np.sum(df_pvals_adj.values <= outlier_threshold, axis=1)
    n_out_median = np.nanmedian(outs_per_sample)
    n_out_max = np.nanmax(outs_per_sample)
    n_out_total = np.nansum(outs_per_sample)
    logger.info(f'Finished computing pvalues. No. outliers per sample in median: {n_out_median}')
    logger.debug(f' {sorted(outs_per_sample)}')

    logger.info(f'Finished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')
    logger.debug(f' {sorted(outs_per_sample)}')

    return Result(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence, df_pvals=df_pvals, df_Z=df_Z,
                  df_pvals_one_sided=df_pvals_one_sided, df_pvals_adj=df_pvals_adj, log2fc=log2fc, fc=fc, n_out_median=n_out_median, n_out_max=n_out_max,
                  n_out_total=n_out_total)
