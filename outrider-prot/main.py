import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path
import pprint
import click

from .model import train, mse_masked
from .datasets import ProtriderDataset
from .stats import get_pvals
from .model_helper import _find_latent_dim, _init_model



#@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
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
def main(config, input_intensities, sample_annotation=None) -> None:
    """# OUTRIDER-PROT

    OUTRIDER-PROT is a package for calling protein outliers on mass spectrometry data

    Links:
    - Publication: FIXME
    - Official code repository: [FIXME]()

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
    elif config['log_func_name'] == "log10":
        log_func = np.log10
    elif config['log_func_name']=='log':
        log_func = np.log
    else:
        raise ValueError(f"Log func {config['log_func_name']} not supported.")

    ## Catch some errors/inconsistencies
    if config['find_q_method']=='OHT' and config['cov_used'] is not None:
        raise ValueError('OHT not implemented with covariate inclusion yet')
    
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
    X_init = model(dataset.X, dataset.cov_one_hot)
    final_loss = mse_masked(dataset.X, X_init,
                            dataset.torch_mask).detach().numpy()
    print('\tInitial loss after model init: ', final_loss )
    
    if not config['autoencoder_training']:
        X_out = X_init
    else:
        print('=== Fitting model ===')
        ## 5. Train model
        final_loss = train(dataset, model, 
              n_epochs = config['n_epochs'],
              learning_rate=float(config['lr']),
             batch_size=config['batch_size']
             )#.detach().numpy()
        X_out = model(dataset.X, dataset.cov_one_hot)
        #final_loss =  mse_masked(dataset.X, X_out, dataset.torch_mask).detach().numpy()
        print('Final loss:', final_loss)
    
    # Store as df
    df_out = pd.DataFrame(X_out.detach().numpy())
    df_out.columns = dataset.data.columns
    df_out.index = dataset.data.index
    
    ## 6. Compute residuals, pvals, zscores
    print('=== Computing statistics ===')
    df_res = dataset.data - df_out # log data - pred data    
    pvals, Z, pvals_adj = get_pvals(df_res.values, 
                         how=config['pval_sided'], 
                         dis=config['pval_dist'],
                         padjust=config["pval_adj"])
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

    pseudocount = config['pseudocount'] #0.01
    log2fc = np.log2(dataset.data + pseudocount) - np.log2(df_out + pseudocount)
    fc = (dataset.data + pseudocount) / (df_out + pseudocount)
        
    
    outs_per_sample = np.sum(df_pvals_adj.values<=config['outlier_threshold'], axis=1)
    
    print(f'\tFinished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')
    print(f'\t {sorted(outs_per_sample)}')
    if config['out_dir'] is not None:
        print('=== Saving output ===')
        out_dir = config['out_dir']
        
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
            
    df_summary = report_summary(dataset.raw_data, dataset.data, df_out, df_Z, 
                               df_pvals, df_pvals_adj, log2fc, fc, 
                                config['pval_dist'], config['outlier_threshold'],
                                config['out_dir'], config['report_all']
                               )
    return df_summary
    #return dataset.data, df_out, df_res, df_pvals, df_Z

def report_summary(raw_in, ae_in, ae_out, zscores,
                   pvals, pvals_adj, 
                   log2fc, fc, 
                   pval_dist='gaussian',
                   outlier_thres=0.1, out_dir=None, include_all=False):
    print('=== Reporting summary ===')
    ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_EXPECTED_LOG2INT'}))
    ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_LOG2INT'}))
    raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_INT'}))
    zscores = (zscores.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_ZSCORE'}))
    pvals = (pvals.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_PVALUE'}))
    pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_PADJ'}))
    log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_LOG2FC'}))
    fc = (fc.reset_index().melt(id_vars='sampleID')
                .rename(columns={'value':'PROTEIN_FC'}))
    
    merge_cols = ['sampleID', 'proteinID']
    df_res = (ae_in.merge(ae_out, on=merge_cols)
                .merge(raw_in, on=merge_cols)
                .merge(zscores, on=merge_cols)
                .merge(pvals, on=merge_cols) 
                .merge(pvals_adj, on=merge_cols)
                .merge(log2fc, on=merge_cols)
                .merge(fc, on=merge_cols)
             ).reset_index(drop=True)

    df_res['PROTEIN_outlier'] = df_res['PROTEIN_PADJ'].apply(lambda x: x<= outlier_thres)
    df_res['pvalDistribution'] = pval_dist

    if not include_all:
        original_len = df_res.shape[0]
        df_res = df_res.query('PROTEIN_outlier==True')
        print(f'\t--- Removing non-significant sample-protein combinations. \n\tOriginal len: {original_len}, new len: {df_res.shape[0]}---')
        
    if out_dir is not None:
        out_p = f'{out_dir}/outrider_prot_summary.csv'
        df_res.to_csv(out_p, index=None)
        print(f'\t--- Wrote output summary with shape {df_res.shape} to <{out_p}>---')
    return df_res
    
    