from collections import defaultdict

import numpy as np
import yaml
from pathlib import Path
import click
import torch
import logging

from .pipeline import run_protrider
from .config import load_config
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
        logging.basicConfig(
            level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

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
    plots.plot_correlation_heatmap(
        out_dir, ctx.obj['sample_annotation'], plot_title, None)


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
        click.echo(
            'No protein_id provided for expected vs observed plot. Exiting.')
        return
    out_dir = ctx.obj['out_dir']
    plot_title = ctx.obj['plot_title']
    logger.info(
        f"plotting expected vs observed protein intensitiy for protein {protein_id}")
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
    plots.plot_correlation_heatmap(
        out_dir, ctx.obj['sample_annotation'], plot_title, covariate_name=covariate)


@cli.command('run')
@click.option(
    "--config",
    help="Configuration file containing PROTRIDER custom options",
    type=click.Path(exists=True, dir_okay=False),
)
def run_cli(config: str):
    """Run the PROTRIDER pipeline.
    """
    run(config)


def run(config_path: str):
    """Run the PROTRIDER pipeline.
    """
    if config_path is None:
        click.echo('No config file provided. Exiting.')
        return

    # Load and validate config using ProtriderConfig dataclass
    config = load_config(config_path)
    
    if config.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

    input_intensities = config.input_intensities
    sample_annotation = config.sample_annotation
    out_dir = config.out_dir

    logger.info('Starting protrider')
    logger.info("Config:\n%s", yaml.dump(config.__dict__, default_flow_style=False))

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Log function and base function are now computed in config.__post_init__
    if config.log_func is None:
        logger.warning(
            'No log function passed for preprocessing. \nAssuming data is already log transformed.')

    if (config.find_q_method == 'OHT') and config.cov_used:
        logger.warning('OHT has not been evaluated with covariates yet')

    if (config.find_q_method == 'OHT') and config.presence_absence:
        logger.warning(
            'OHT has not been evaluated on presence/absence analysis yet')

    if config.seed is not None:
        logger.info('Setting random seed: %s', config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # Run PROTRIDER
    result, model_info = run_protrider(input_intensities, config, sample_annotation)

    # Save results
    # Save wide format (individual CSV files)
    result.save(out_dir, format="wide")
    # Save long format summary
    result.save(out_dir, format="long", include_all=config.report_all)
    # Save model information
    model_info.save(out_dir)
    # Save config
    config.save(out_dir)


if __name__ == '__main__':
    cli(obj={})
