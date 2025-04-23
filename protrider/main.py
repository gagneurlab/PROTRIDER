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
    "--input_intensities",
    help="csv input file containing intensities. Columns are samples and rows are proteins. See example here: FIXME",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--sample_annotation",
    help="csv file containing sample annotations",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out_dir",
    help="Output directory to save results",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
)
def main(config, input_intensities: str, sample_annotation: str = None, out_dir: str = None) -> None:
    """# PROTRIDER

    PROTRIDER is a package for calling protein outliers on mass spectrometry data

    Links:
    - Publication: FIXME
    - Official code repository: https://github.com/gagneurlab/PROTRIDER

    """

    ## Load config with params
    config = yaml.load(open(config), Loader=yaml.FullLoader)

    if config['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', )

    logger.info('Starting protrider')
    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False))

    config = defaultdict(lambda: None, config)

    if out_dir is not None:
        config['out_dir'] = out_dir

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
            logger.warning('No log function passed for preprocessing. \nAssuming data is already log transformed.')
            log_func = lambda x: x  # id
            base_fn = np.exp
        else:
            raise ValueError(f"Log func {config['log_func_name']} not supported.")

    ## Catch some errors/inconsistencies
    if config['find_q_method'] == 'OHT' and config['cov_used'] is not None:
        raise ValueError('OHT not implemented with covariate inclusion yet')

    device = torch.device("cuda" if ((torch.cuda.is_available()) & (config['device'] == 'gpu')) else "cpu")

    if config.get('seed', None) is not None:
        logger.info('Setting random seed: %s', config['seed'])
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    if config.get('cross_val', False):
        if config['find_q_method'] != 'OHT':
            raise ValueError('Cross-validation only implemented with OHT method')
        result, model_info, df_folds = run_experiment_cv(input_intensities, config, sample_annotation,
                                                         log_func, base_fn, device=device)
    else:
        result, model_info, = run_experiment(input_intensities, config, sample_annotation, log_func, base_fn,
                                             device=device)
        df_folds = None

    summary = _report_summary(result, config['pval_dist'], config['outlier_threshold'],
                              config['report_all'])

    if config['out_dir'] is not None:
        _write_results(summary=summary, result=result, model_info=model_info, out_dir=config['out_dir'],
                       df_folds=df_folds, config=config)

    return summary


def _write_results(summary, result: Result, model_info: ModelInfo, out_dir, config: dict, df_folds: DataFrame = None):
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

    # p-values
    out_p = f'{out_dir}/pvals.csv'
    result.df_pvals.T.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved P-values to {out_p}")

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
    df_info = pd.DataFrame(model_info_dict, index=pd.Index(folds, name='fold'))
    df_info.to_csv(out_p, header=True, index=True)
    logger.info(f"Saved additional input to {out_p}")

    # folds
    if df_folds is not None:
        out_p = f'{out_dir}/folds.csv'
        df_folds.to_csv(out_p, header=True, index=True)
        logger.info(f"Saved folds to {out_p}")

    out_p = f'{out_dir}/protrider_summary.csv'
    summary.to_csv(out_p, index=None)
    logger.info(f'Saved output summary with shape {summary.shape} to <{out_p}>---')


def _report_summary(result: Result, pval_dist='gaussian', outlier_thres=0.1, include_all=False):
    ae_out = result.df_out
    ae_in = result.dataset.data
    raw_in = result.dataset.raw_data
    zscores = result.df_Z
    pvals = result.df_pvals
    pvals_adj = result.df_pvals_adj
    log2fc = result.log2fc
    fc = result.fc

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
