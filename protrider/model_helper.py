import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import torch
import copy

from .stats import get_pvals, fit_residuals
from .model import ProtriderAutoencoder, train, MSEBCELoss  # masked
from .datasets import ProtriderSubset
import logging

__all__ = ['init_model', 'find_latent_dim']

logger = logging.getLogger(__name__)


def find_latent_dim(dataset, method='OHT',
                    inj_freq=1e-3, inj_mean=3, inj_sd=1.6,
                    init_wPCA=True, n_layers=1, h_dim=None,
                    n_epochs=100, learning_rate=1e-6, batch_size=None,
                    pval_sided='two-sided', pval_dist='gaussian',
                    out_dir=None, device=torch.device('cpu'),
                    presence_absence=False, lambda_bce=1.
                    ):
    if method == "OHT" or method == "oht":
        logger.info('OHT method for finding latent dim')
        dataset.perform_svd()
        q = dataset.find_enc_dim_optht()
    elif method == "gs":
        logger.info('Grid search method for finding latent dim')
        logger.info('Injecting outliers')
        inj_freq = float(inj_freq)
        learning_rate = float(learning_rate)
        injected_dataset, outlier_mask = _inject_outliers(dataset, inj_freq, inj_mean, inj_sd, device=device)

        possible_qs = _get_gs_params(dataset.X.shape)
        logger.info("Starting grid search for optimal encoding dimension")
        gridSearch_results = []
        for latent_dim in possible_qs:
            logger.info(f"Testing q = {latent_dim}")
            model = init_model(injected_dataset, latent_dim, init_wPCA, n_layers, h_dim, device,
                               presence_absence=presence_absence)
            criterion = MSEBCELoss(presence_absence=presence_absence, lambda_bce=lambda_bce)
            X_out = model(injected_dataset.X, injected_dataset.torch_mask, cond=injected_dataset.cov_one_hot)
            loss, mse_loss, bce_loss = criterion(X_out, injected_dataset.X, injected_dataset.torch_mask, detached=True)
            logger.info('\tInitial loss after model init: %s, mse_loss: %s, bce_loss: %s',
                        loss, mse_loss, bce_loss)

            logger.info('\tFitting model')
            loss, mse_loss, bce_loss = train(injected_dataset, model, criterion, n_epochs, learning_rate, batch_size)
            logger.info('\tFinal loss after model fit: %s, mse_loss: %s, bce_loss: %s',
                        loss, mse_loss, bce_loss)
            X_out = model(injected_dataset.X, injected_dataset.torch_mask,
                          cond=injected_dataset.cov_one_hot).detach().cpu().numpy()
            if presence_absence:
                presence_out = X_out[1]
                X_out = X_out[0]

            if ~np.isfinite(loss):
                auc_prec_rec = np.nan
            else:
                X_in = copy.deepcopy(injected_dataset.X).detach().cpu().numpy()
                X_in[injected_dataset.mask] = np.nan
                res = X_in - X_out
                mu, sigma, df0 = fit_residuals(X_in - X_out, dis='gaussian')
                pvals, _ = get_pvals(res,
                                     mu=mu, sigma=sigma, df0=df0,
                                     how=pval_sided,
                                     dis='gaussian',
                                     )
                auprc = _get_prec_recall(pvals, outlier_mask)
                logger.info(f"\t==> q = {latent_dim}: AUCPR = {auprc}")
                gridSearch_results.append([latent_dim, auprc])

        df_gs = pd.DataFrame(gridSearch_results, columns=["encod_dim", "aucpr"])
        q = int(df_gs.loc[df_gs['aucpr'].idxmax()]['encod_dim'])
        logger.info(f'Finished grid search. Optimal encoding dimension = {q}.')

        if out_dir is not None:
            out_p = f'{out_dir}/grid_search.csv'
            df_gs.to_csv(out_p, header=True, index=True)
            logger.info(f"\t Saved grid_search to {out_p}")
    else:
        print("Setting q is a fixed user-provided value")
        q = int(method)
    return q


def init_model(dataset, latent_dim, init_wPCA=True, n_layer=1, h_dim=None, device=torch.device('cpu'),
               presence_absence=False):
    n_cov = dataset.cov_one_hot.shape[1]
    n_prots = dataset.X.shape[1]
    model = ProtriderAutoencoder(in_dim=n_prots, latent_dim=latent_dim, n_layers=n_layer, h_dim=h_dim, n_cov=n_cov,
                                 prot_means=None if init_wPCA else dataset.prot_means_torch,
                                 presence_absence=presence_absence)
    model.double().to(device)

    if init_wPCA:
        logger.info('\tInitializing model weights with PCA')
        dataset.perform_svd()
        Vt_q = dataset.Vt[:latent_dim]
        model.initialize_wPCA(Vt_q, dataset.prot_means, n_cov)
    return model


def _get_gs_params(data_shape, MP=2, a=3, max_steps=30):
    b = round(min(data_shape) / MP)
    n_steps = min(max_steps, b)  # do at most 25 steps or N/3
    par_q = np.unique(np.round(np.exp(np.linspace(start=np.log(a),
                                                  stop=np.log(b),
                                                  num=n_steps))))
    return par_q.astype(int).tolist()


def _rlnorm(size, inj_mean, inj_sd):
    log_mean = np.log(inj_mean) if inj_mean != 0 else 0
    return np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)


def _inject_outliers(dataset, inj_freq=1e-3, inj_mean=3, inj_sd=1.6, device=torch.device('cpu')):
    max_outlier_value = np.nanmin([100 * np.nanmax(dataset.X.detach().cpu().numpy()),
                                   torch.finfo(dataset.X.dtype).max])
    logger.info('max value %s', max_outlier_value)
    X_prepro = dataset.X  ## uncentered data without NAs
    X_trans = dataset.X

    # draw where to inject
    outlier_mask = np.random.choice(
        [0., -1., 1.], size=dataset.X.shape,
        p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])

    # insert with log normally distributed zscore in transformed space
    inj_zscores = _rlnorm(size=dataset.X.shape, inj_mean=inj_mean, inj_sd=inj_sd)
    sd = np.nanstd(X_trans.detach().cpu(), ddof=1, axis=0)
    # sd = torch.nanstd(X_trans, dim=0, unbiased=True)

    # reverse transform to original space
    X_injected = torch.tensor(outlier_mask * inj_zscores * sd).to(device) + dataset.X

    # avoid inj outlier to be too strong
    cond = X_injected > max_outlier_value
    X_injected[cond] = dataset.X[cond]
    outlier_mask[cond.detach().cpu()] = 0

    # set original NA to NAs in injected dataset
    outlier_mask[dataset.mask] = np.nan
    X_injected[dataset.mask] = np.nan
    nr_out = np.sum(np.abs(outlier_mask[np.isfinite(outlier_mask)]))
    logger.info(f"Injecting {nr_out} outliers (freq = {nr_out / dataset.X.nelement()})")

    if isinstance(dataset, ProtriderSubset):
        ds_injected = dataset.deepcopy_to_dataset()
    else:
        ds_injected = copy.deepcopy(dataset)

    # ds_injected.X = X_injected # why not just this?
    ds_injected.X = torch.where(dataset.torch_mask, dataset.X, X_injected).to(device)
    ds_injected.prot_means = np.nanmean(ds_injected.X.detach().cpu(), axis=0, keepdims=1)

    ## PCA should be computed on ds_injected.X - prot_means
    ds_injected.centered_log_data_noNA = ds_injected.X.detach().cpu().numpy() - ds_injected.prot_means
    return ds_injected, outlier_mask


def _get_prec_recall(X_pvalue, X_is_outlier):
    score = -X_pvalue[~np.logical_or(np.isnan(X_pvalue),
                                     np.isnan(X_is_outlier))]
    label = X_is_outlier[~np.logical_or(np.isnan(X_pvalue),
                                        np.isnan(X_is_outlier))]

    if np.sum(np.abs(label)) == 0:
        warnings.warn("no injected outliers found"
                      " -> no precision-recall calculation possible")

    label = (label != 0).astype('int')

    pre, rec, _ = precision_recall_curve(label, score)
    curve_auc = auc(rec, pre)
    return curve_auc  # {"auc": curve_auc, "pre": pre, "rec": rec}
