import dataclasses
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import click
from pandas import DataFrame
import torch
import logging
import os

from .utils import Result, ModelInfo, run_experiment, run_experiment_cv
from .plots import _plot_pvals, _plot_encoding_dim, _plot_aberrant_per_sample, _plot_training_loss, _plot_expected_vs_observed

logger = logging.getLogger(__name__)


# @click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.command()
@click.option(
    "--config",
    default='../config.yaml',
    help="Configuration file containing PROTRIDER custom options",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    '--run_pipeline',
    is_flag=True,
    help="Run PROTRIDER pipeline"
)
@click.option(
    '--plot_heatmap',
    is_flag=True,
    help="Plot the correlation heatmaps"
)
@click.option(
    '--plot_title',
    type=str,
    default="",
    help="Title of the plots"
)
@click.option(
    '--plot_pvals',
    is_flag=True,
    help="Plot the pvalue plots"
)
@click.option(
    '--plot_encoding_dim',
    is_flag=True,
    help="Plot the endocing dimension search plot"
)
@click.option(
    '--plot_aberrant_per_sample',
    is_flag=True,
    help="Plot nubmer of aberrant proteins per sample"
)
@click.option(
    '--plot_training_loss',
    is_flag=True,
    help="Plot training loss history"
)
@click.option(
    '--plot_expected_vs_observed',
    is_flag=True,
    help="Plot expected vs observed protein intensitiy for protein protein_id"
)
@click.option(
    '--protein_id',
    type=str,
    help="Id of the protein to plot"
)
@click.option(
    '--plot_all',
    is_flag=True,
    help="Plot training loss, nubmer of aberrant proteins per sample, pvalue plots and endocing dimension search"
)
def main(config, run_pipeline: bool = False, plot_heatmap: bool = False, plot_title: str = "", plot_pvals: bool = False, 
         plot_encoding_dim: bool = False, plot_aberrant_per_sample: bool = False, plot_training_loss: bool = False, 
         plot_expected_vs_observed: bool = False, protein_id: str = None, plot_all: bool = False) -> None:
    """# PROTRIDER

    PROTRIDER is a package for calling protein outliers on mass spectrometry data

    Links:
    - Publication: FIXME
    - Official code repository: https://github.com/gagneurlab/PROTRIDER

    """
    print(plot_title)

    return run(config, input_intensities, sample_annotation, out_dir)


def run(config, input_intensities: str, sample_annotation: str = None, out_dir: str = None, skip_summary=False):
    ## Load config with params
    config = yaml.load(open(config), Loader=yaml.FullLoader) 
    if run_pipeline is True:
        logger.info('Runing PROTRIDER pipeline')
        return run(config)
    elif plot_heatmap is True:
        # TODO add plot_heatmap
        logger.info("plotting correlation_heatmaps is not implemented yet.")
        return
    elif plot_pvals is True:
        logger.info("plotting pvalue plots")
        _plot_pvals(config["out_dir"], config['pval_dist'], plot_title)
    elif plot_encoding_dim is True:
         logger.info("plotting encoding dimension search plot")
         _plot_encoding_dim(config["out_dir"], config['find_q_method'], plot_title)
    elif plot_aberrant_per_sample is True:
        logger.info("plotting number of aberrant proteins per sample")
        _plot_aberrant_per_sample(config["out_dir"], plot_title)
    elif plot_training_loss is True:
        logger.info("plotting training loss")
        _plot_training_loss(config["out_dir"], plot_title)
    elif plot_expected_vs_observed is True:
        if protein_id is None:
            raise ValueError("protein_id is required for plot_expected_vs_observed function.")
        logger.info(f"plotting expected vs observed protein intensitiy for protein {protein_id}")
        _plot_expected_vs_observed(config["out_dir"], protein_id, plot_title)
    elif plot_all is True:
        _plot_pvals(config["out_dir"], config['pval_dist'], plot_title)
        _plot_aberrant_per_sample(config["out_dir"], plot_title)
        _plot_encoding_dim(config["out_dir"], config['find_q_method'], plot_title)
        _plot_training_loss(config["out_dir"], plot_title)
        if protein_id is None:
            raise ValueError("protein_id is required for plot_expected_vs_observed function.")
        _plot_expected_vs_observed(config["out_dir"], protein_id, plot_title)


def run(config):
    if config['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

    logger.info('Starting protrider')
    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False))

    config = defaultdict(lambda: None, config)

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
            logger.warning('No log function passed for preprocessing. \nAssuming data is already log transformed.')
            log_func = None
            base_fn = np.exp
        else:
            raise ValueError(f"Log func {config['log_func_name']} not supported.")

    if (config['find_q_method'] == 'OHT') and (len(config.get('cov_used', [])) > 0):
        logger.warning('OHT has not been evaluated with covariates yet')

    if (config['find_q_method'] == 'OHT') and config['presence_absence']:
        logger.warning('OHT has not been evaluated on presence/absence analysis yet')
        
    #if (config['presence_absence'] == True) and (config['n_layers']!=1):
    #    raise ValueError('Presence absence inclusion is only with 1-layers models possible')

    device = torch.device("cuda" if ((torch.cuda.is_available()) & (config['device'] == 'gpu')) else "cpu")

    if config.get('seed', None) is not None:
        logger.info('Setting random seed: %s', config['seed'])
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    if config.get('cross_val', False):
        result, model_info, df_folds = run_experiment_cv(config, log_func, base_fn, device=device)
    else:
        result, model_info, = run_experiment(config, log_func, base_fn, device=device)
        df_folds = None

    if config['out_dir'] is not None:
        _write_results(result=result, model_info=model_info, out_dir=config['out_dir'],
                       df_folds=df_folds, config=config)

    if skip_summary:
        return None
    
    summary = _report_summary(result, config['pval_dist'], config['outlier_threshold'],
                              config['report_all'])
    summary_p = f"{config['out_dir']}/protrider_summary.csv"
    summary.to_csv(summary_p, index=None)
    logger.info(f'Saved output summary with shape {summary.shape} to <{summary_p}>---')

    return summary


def _write_results(result: Result, model_info: ModelInfo, out_dir, config: dict, df_folds: DataFrame = None):
    logger.info('=== Saving output ===')
    out_dir = out_dir

    # AE input
    out_p = f'{out_dir}/processed_input.csv'
    result.dataset.data.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved processed_input to {out_p}")

    # AE output
    out_p = f'{out_dir}/output.csv'
    result.df_out.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved output to {out_p}")

    # residuals
    out_p = f'{out_dir}/residuals.csv'
    result.df_res.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved residuals to {out_p}")

    if result.df_presence is not None:
        out_p = f'{out_dir}/presence_probs.csv'
        result.df_presence.T.to_csv(out_p, header=True, index=True)
        logger.info(f"Saved presence probabilities to {out_p}")
    
    # p-values
    out_p = f'{out_dir}/pvals.csv'
    result.df_pvals.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved P-values to {out_p}")
    
    # left-sided p-values
    out_p = f'{out_dir}/pvals_one_sided.csv'
    result.df_pvals_one_sided.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved left-sided P-values to {out_p}")

    # p-values adj
    out_p = f'{out_dir}/pvals_adj.csv'
    result.df_pvals_adj.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved adjusted P-values to {out_p}")

    # Z-scores
    out_p = f'{out_dir}/zscores.csv'
    result.df_Z.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved z scores to {out_p}")

    # log2fc
    out_p = f'{out_dir}/log2fc.csv'
    result.log2fc.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved log2fc scores to {out_p}")

    # fc
    out_p = f'{out_dir}/fc.csv'
    result.fc.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved fc scores to {out_p}")

    # config
    out_p = f'{out_dir}/config.yaml'
    with open(out_p, 'w') as f:
        yaml.safe_dump(dict(config), f)
    logger.info(f"Saved run config to {out_p}")

    # latent space
    # FIXME

    ## TODO check with master branch additional info

    # Additional info
    out_p = f'{out_dir}/additional_info.csv'
    model_info_dict = dataclasses.asdict(model_info)
    if model_info.q.ndim == 0:
        # make all variables of model_info arrays
        model_info_dict = {k: np.array([v]) for k, v in model_info_dict.items()}

    folds = np.arange(len(model_info_dict['q']))
    if df_folds is None:
        train_losses = model_info_dict.pop("train_losses")
        train_losses_df = pd.DataFrame({
            'epoch': range(1, len(train_losses[0]) + 1),
            'train_loss': train_losses[0],
        }) 
        out_p = f'{out_dir}/train_losses.csv'
        train_losses_df.to_csv(out_p, header=True, index=False)
        logger.info(f"Saved training losses to {out_p}")
    
    out_p = f'{out_dir}/additional_info.csv'
    df_info = pd.DataFrame(model_info_dict, index=pd.Index(folds, name='fold'))
    df_info.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved additional input to {out_p}")

    # folds
    if df_folds is not None:
        out_p = f'{out_dir}/folds.csv'
        df_folds.to_csv(out_p, header=True, index=True)
        logger.info(f"Saved folds to {out_p}")



def _report_summary(result: Result, pval_dist='gaussian', outlier_thres=0.1, include_all=False):
    ae_out = result.df_out
    ae_in = result.dataset.data
    raw_in = result.dataset.raw_data
    zscores = result.df_Z
    pvals = result.df_pvals
    pvals_adj = result.df_pvals_adj
    log2fc = result.log2fc
    fc = result.fc
    presence = result.df_presence

    logger.info('=== Reporting summary ===')
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

    if presence is not None:
        presence = (presence.reset_index().melt(id_vars='sampleID')
              .rename(columns={'value': 'pred_presence_probability'}))
        df_res = df_res.merge(presence, on=merge_cols).reset_index(drop=True)
        
    df_res['PROTEIN_outlier'] = df_res['PROTEIN_PADJ'].apply(lambda x: x <= outlier_thres)
    df_res['pvalDistribution'] = pval_dist

    if not include_all:
        original_len = df_res.shape[0]
        df_res = df_res.query('PROTEIN_outlier==True')
        logger.info(
            f'\t--- Removing non-significant sample-protein combinations. \n\tOriginal len: {original_len}, new len: {df_res.shape[0]}---')

    return df_res


if __name__ == '__main__':
    main()
