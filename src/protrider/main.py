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

from .utils import Result, ModelInfo, run_experiment, run_experiment_cv
from . import plots

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """# PROTRIDER

    PROTRIDER is a package for calling protein outliers on mass spectrometry data

    Links:
    - Publication: FIXME
    - Official code repository: https://github.com/gagneurlab/PROTRIDER

    """
    pass

@cli.group(chain=True)
@click.option(
    "--config",
    help="Configuration file containing PROTRIDER custom options",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    '--plot_title',
    type=str,
    default="",
    help="Title of the plots"
)
@click.pass_context
def plot(ctx, config: str, plot_title: str = ""):
    """
    Plot the results of a PROTRIDER run.
    """
    if config is None:
        click.echo('No config file provided. Exiting.')
        return
    
    config = yaml.load(open(config), Loader=yaml.FullLoader) 
    config = defaultdict(lambda: None, config)
    if config['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

    out_dir = config['out_dir']
    if Path(out_dir).is_dir() is False:
        click.echo(f'Output directory {out_dir} does not exist. Exiting.')
        return

    ctx.obj = config
    ctx.obj['plot_title'] = plot_title


@plot.command('all')
@click.pass_context
def plot_all(ctx):
    """
    Plot all results for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting all plots")
    plots.plot_pvals(out_dir, ctx.obj['pval_dist'], plot_title)
    plots.plot_aberrant_per_sample(out_dir, plot_title)
    plots.plot_encoding_dim(out_dir, ctx.obj['find_q_method'], plot_title)
    plots.plot_training_loss(out_dir, plot_title)
    plots.plot_correlation_heatmap(out_dir, ctx.obj['sample_annotation'], plot_title, None)

@plot.command('pvals')
@click.pass_context
def plot_pvals(ctx):
    """
    Plot pvalue plots for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting pvalue plots")
    plots.plot_pvals(out_dir, ctx.obj['pval_dist'], plot_title)

@plot.command('aberrant_per_sample')
@click.pass_context
def plot_aberrant_per_sample(ctx):
    """
    Plot number of aberrant proteins per sample for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting number of aberrant proteins per sample")
    plots.plot_aberrant_per_sample(out_dir, plot_title)

@plot.command('encoding_dim')
@click.pass_context
def plot_encoding_dim(ctx):
    """
    Plot encoding dimension search plot for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting encoding dimension search plot")
    plots.plot_encoding_dim(out_dir, ctx.obj['find_q_method'], plot_title)

@plot.command('training_loss')
@click.pass_context
def plot_training_loss(ctx):
    """
    Plot training loss history for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting training loss")
    plots.plot_training_loss(out_dir, plot_title)

@plot.command('expected_vs_observed')
@click.option(
    '--protein_id',
    type=str,
    help="Id of the protein to plot"
)
@click.pass_context
def plot_expected_vs_observed(ctx, protein_id: str):
    """
    Plot expected vs observed protein intensity for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    if protein_id is None:
        click.echo('No protein_id provided for expected vs observed plot. Exiting.')
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info(f"plotting expected vs observed protein intensitiy for protein {protein_id}")
    plots.plot_expected_vs_observed(out_dir, protein_id, plot_title)   

@plot.command('correlation_heatmap')
@click.option(
    '--covariate',
    type=str,
    help="Name of the covariate to color the samples by"
)
@click.pass_context
def plot_correlation_heatmap(ctx, covariate: str):
    """
    Plot correlation heatmap for a PROTRIDER run.
    """
    if ctx.obj is None:
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info("plotting correlation heatmap")
    plots.plot_correlation_heatmap(out_dir, ctx.obj['sample_annotation'], plot_title, covariate_name=covariate)

@cli.command()
@click.option(
    "--config",
    help="Configuration file containing PROTRIDER custom options",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--skip_summary",
    help="Do not save the summary file.",
    is_flag=True,
)
def run(config: str, skip_summary: bool = False) -> DataFrame:
    """Run the PROTRIDER pipeline.
    """
    if config is None:
        click.echo('No config file provided. Exiting.')
        return
    
    config = yaml.load(open(config), Loader=yaml.FullLoader) 
    config = defaultdict(lambda: None, config)
    if config['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

    input_intensities = config['input_intensities']
    sample_annotation = config['sample_annotation']
    out_dir = config['out_dir']

    logger.info('Starting protrider')
    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False))

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    if config['log_func_name'] == "log2":
        log_func = np.log2
        def base_fn(x):
            return 2 ** x
    elif config['log_func_name'] == "log10":
        log_func = np.log10
        def base_fn(x):
            return 10 ** x
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

    if (config['find_q_method'] == 'OHT') and config['cov_used']:
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
        result, model_info, df_folds = run_experiment_cv(input_intensities, config, sample_annotation,
                                                         log_func, base_fn, device=device, out_dir=out_dir)
    else:
        result, model_info, = run_experiment(input_intensities, config, sample_annotation, log_func, base_fn,
                                             device=device, out_dir=out_dir)
        df_folds = None

    _write_results(result=result, model_info=model_info, out_dir=out_dir,
                    df_folds=df_folds, config=config)
        
    if not skip_summary:
        summary = _report_summary(result, config['pval_dist'], config['outlier_threshold'],
                                config['report_all'])
        summary_p = f"{out_dir}/protrider_summary.csv"
        summary.to_csv(summary_p, index=None)
        logger.info(f'Saved output summary with shape {summary.shape} to <{summary_p}>---')

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
    cli(obj={})