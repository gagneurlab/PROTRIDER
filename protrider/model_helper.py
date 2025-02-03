import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import torch
import copy

from .stats import get_pvals
from .model import ProtriderAutoencoder, train, mse_masked

def _find_latent_dim(dataset, method='OHT', 
                     inj_freq=1e-3, inj_mean=3, inj_sd=1.6, seed=None,
                     init_wPCA=True, n_layers=1, h_dim=None, 
                     n_epochs=100, learning_rate=1e-6, batch_size=None,
                     pval_sided='two-sided', pval_dist='gaussian', 
                     out_dir=None
                    ):
    if method=="OHT":
        print('-- OHT method for finding latent dim ---')
        dataset._perform_svd()
        q = dataset._find_enc_dim_optht()
    else:
        print('--- Grid search method for finding latent dim ---')
        print('--- Injecting outliers ---')
        injected_dataset, outlier_mask = _inject_outliers(dataset, inj_freq, inj_mean, inj_sd, seed)     

        possible_qs = _get_gs_params(dataset.X.shape)
        print("--- Starting grid search for optimal encoding dimension ---")
        gridSearch_results = []
        for latent_dim in possible_qs:
            print(f"--- Testing q = {latent_dim} ---")
            model = _init_model(injected_dataset, latent_dim, init_wPCA, n_layers, h_dim)
            X_init = model(injected_dataset.X, 
                           prot_means=injected_dataset.prot_means_torch,
                           cond=injected_dataset.cov_one_hot)
            final_loss = mse_masked(injected_dataset.X, X_init,
                                    injected_dataset.torch_mask).detach().numpy()
            print('\tInitial loss after model init: ', final_loss )
            
            print('\t--- Fitting model ---')
            final_loss = train(injected_dataset, model, 
                               n_epochs, learning_rate, batch_size)
            print(f'\tFinal loss: {final_loss}')
            
            X_out = model(injected_dataset.X,  
                          prot_means=injected_dataset.prot_means_torch,
                          cond=injected_dataset.cov_one_hot).detach().cpu().numpy()
        
            if ~np.isfinite(final_loss):
                auc_prec_rec = np.nan
            else:
                X_in = copy.deepcopy(injected_dataset.X).detach().numpy()
                X_in[injected_dataset.mask] = np.nan
                pvals, _, _ = get_pvals(X_in - X_out, 
                                     how=pval_sided, 
                                     dis='gaussian',
                                     padjust=None
                                    )
                auprc = _get_prec_recall(pvals, outlier_mask)
                print(f"\t==> q = {latent_dim}: AUCPR = {auprc}")
                gridSearch_results.append([latent_dim, auprc])
        
        df_gs = pd.DataFrame( gridSearch_results, columns=["encod_dim", "aucpr"])
        q = int(df_gs.loc[df_gs['aucpr'].idxmax()]['encod_dim'])
        print(f'--- Finished grid search. Optimal encoding dimension = {q}. ---')
        
        if out_dir is not None:
            out_p = f'{out_dir}/grid_search.csv'
            df_gs.to_csv(out_p, header=True, index=True)
            print(f"\t Saved grid_search to {out_p}")
        
    return q


def _init_model(dataset, latent_dim, init_wPCA = True, n_layer=1, h_dim=None):
    n_cov = dataset.cov_one_hot.shape[1]
    n_prots = dataset.X.shape[1] 
    model = ProtriderAutoencoder(in_dim=n_prots, latent_dim=latent_dim, 
                                 n_layers=n_layer, h_dim=h_dim,
                                 n_cov=n_cov
                                )
    model.double()
    
    if init_wPCA:
        print('\tInitializing model weights with PCA')
        dataset._perform_svd()
        Vt_q = dataset.Vt[:latent_dim]
        model._initialize_wPCA(Vt_q, dataset.prot_means, n_cov)
    return model


def _get_gs_params(data_shape, MP=2, a=4, max_steps=25):
    b = round(min(data_shape) / MP)
    n_steps = min(max_steps, b)  # do at most 25 steps or N/3
    par_q = np.unique(np.round(np.exp(np.linspace(start=np.log(a),
                                                    stop=np.log(b),
                                                    num=n_steps))))
    return par_q.astype(int).tolist()


def _rlnorm(size, inj_mean, inj_sd):
    log_mean = np.log(inj_mean) if inj_mean != 0 else 0
    return np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)

    
def _inject_outliers(dataset, inj_freq=1e-3, inj_mean=3, inj_sd=1.6, seed=None):
    max_outlier_value = np.nanmin([100 * np.nanmax(dataset.X),
                                   torch.finfo(dataset.X.dtype).max])
    print('max value', max_outlier_value)
    X_prepro = dataset.X ## uncentered data without NAs
    X_trans = dataset.X
    
    # draw where to inject
    if seed is not None:
        np.random.seed(seed)
    outlier_mask = np.random.choice(
                        [0., -1., 1.], size=dataset.X.shape,
                        p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])
    
    # insert with log normally distributed zscore in transformed space
    inj_zscores = _rlnorm(size=dataset.X.shape, inj_mean=inj_mean, inj_sd=inj_sd)
    sd = np.nanstd(X_trans, ddof=1, axis=0)
    
    # reverse transform to original space
    X_injected = torch.tensor(outlier_mask * inj_zscores * sd) + dataset.X
    
    # avoid inj outlier to be too strong
    cond = X_injected > max_outlier_value
    X_injected[cond] = dataset.X[cond]
    outlier_mask[cond] = 0

    # set original NA to NAs in injected dataset
    outlier_mask[dataset.mask] = np.nan
    X_injected[dataset.mask] = np.nan
    nr_out = np.sum(np.abs(outlier_mask[np.isfinite(outlier_mask)]))
    print(f"Injecting {nr_out} outliers (freq = {nr_out/dataset.X.nelement()})")

    ds_injected = copy.deepcopy(dataset)
    #ds_injected.X = X_injected # why not just this?
    ds_injected.X = torch.where(dataset.torch_mask, dataset.X, X_injected)
    ds_injected.prot_means = np.nanmean(ds_injected.X, axis=0, keepdims=1)

    ## PCA should be computed on ds_injected.X - prot_means
    ds_injected.centered_log_data_noNA  = ds_injected.X - ds_injected.prot_means
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
    return curve_auc #{"auc": curve_auc, "pre": pre, "rec": rec}