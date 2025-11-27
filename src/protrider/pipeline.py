import numpy as np
import pandas as pd
from typing import Union, Tuple, Literal, Optional
import logging
import torch
from dataclasses import dataclass

from .model import train, train_val, MSEBCELoss, ProtriderAutoencoder, find_latent_dim, init_model, ModelInfo
from .datasets import ProtriderDataset, ProtriderSubset, ProtriderKfoldCVGenerator, ProtriderLOOCVGenerator
from .stats import get_pvals, fit_residuals, adjust_pvals
from .plots import plot_cv_loss
from .config import ProtriderConfig


__all__ = ["run"]

logger = logging.getLogger(__name__)

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
    pval_dist: str = 'gaussian'  # Distribution used for p-value computation
    outlier_threshold: float = 0.1  # Threshold for determining outliers
    
    def save(self, out_dir: str, format: Literal["wide", "long"] = "wide", 
             include_all: bool = False) -> Optional[pd.DataFrame]:
        """
        Save result dataframes to CSV files.
        
        Args:
            out_dir: Output directory path where files will be saved
            format: Output format - "wide" saves separate CSV files for each metric,
                   "long" saves a single combined CSV in long format
            include_all: If False, only save significant results in long format
                        (uses self.outlier_threshold for filtering)
            
        Returns:
            DataFrame if format="long", None if format="wide"
            
        """
        if format == "wide":
            logger.info('=== Saving results in wide format ===')
            
            # AE input
            out_p = f'{out_dir}/processed_input.csv'
            self.dataset.data.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved processed_input to {out_p}")

            # AE output
            out_p = f'{out_dir}/output.csv'
            self.df_out.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved output to {out_p}")

            # residuals
            out_p = f'{out_dir}/residuals.csv'
            self.df_res.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved residuals to {out_p}")

            if self.df_presence is not None:
                out_p = f'{out_dir}/presence_probs.csv'
                self.df_presence.T.to_csv(out_p, header=True, index=True)
                logger.info(f"Saved presence probabilities to {out_p}")

            # p-values
            out_p = f'{out_dir}/pvals.csv'
            self.df_pvals.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved P-values to {out_p}")

            # left-sided p-values
            out_p = f'{out_dir}/pvals_one_sided.csv'
            self.df_pvals_one_sided.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved left-sided P-values to {out_p}")

            # p-values adj
            out_p = f'{out_dir}/pvals_adj.csv'
            self.df_pvals_adj.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved adjusted P-values to {out_p}")

            # Z-scores
            out_p = f'{out_dir}/zscores.csv'
            self.df_Z.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved z scores to {out_p}")

            # log2fc
            out_p = f'{out_dir}/log2fc.csv'
            self.log2fc.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved log2fc scores to {out_p}")

            # fc
            out_p = f'{out_dir}/fc.csv'
            self.fc.T.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved fc scores to {out_p}")
            
            return None
            
        elif format == "long":
            logger.info('=== Saving results in long format ===')
            
            ae_out = self.df_out
            ae_in = self.dataset.data
            raw_in = self.dataset.raw_data
            zscores = self.df_Z
            pvals = self.df_pvals
            pvals_adj = self.df_pvals_adj
            log2fc = self.log2fc
            fc = self.fc
            presence = self.df_presence

            ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                      .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_EXPECTED_LOG2INT'}))
            ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2INT'}))
            raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                      .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_INT'}))
            zscores = (zscores.reset_index().melt(id_vars='sampleID')
                       .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_ZSCORE'}))
            pvals = (pvals.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PVALUE'}))
            pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                         .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PADJ'}))
            log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                      .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2FC'}))
            fc = (fc.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_FC'}))

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
                            .rename(columns={'variable': 'proteinID', 'value': 'pred_presence_probability'}))
                df_res = df_res.merge(presence, on=merge_cols).reset_index(drop=True)

            df_res['PROTEIN_outlier'] = df_res['PROTEIN_PADJ'].apply(
                lambda x: x <= self.outlier_threshold)
            df_res['pvalDistribution'] = self.pval_dist

            if not include_all:
                original_len = df_res.shape[0]
                df_res = df_res.query('PROTEIN_outlier==True')
                logger.info(
                    f'\t--- Removing non-significant sample-protein combinations. \n\tOriginal len: {original_len}, new len: {df_res.shape[0]}---')

            out_p = f"{out_dir}/protrider_summary.csv"
            df_res.to_csv(out_p, index=None)
            logger.info(f'Saved output summary with shape {df_res.shape} to {out_p}')
            
            return df_res
    
    def plot_pvals(self, out_dir: str = None, **kwargs):
        """
        Plot p-value distributions.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plots are returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize, distribution)
            
        Returns:
            tuple: (histogram_plot, qq_plot) - plotnine plot objects
            
        Example:
            >>> hist, qq = result.plot_pvals()  # Interactive use
            >>> hist.draw()
            >>> result.plot_pvals(out_dir='output/')  # Save plots
        """
        from . import plots
        # Pass the one-sided p-values DataFrame
        pvals_one_sided = self.df_pvals_one_sided if hasattr(self, 'df_pvals_one_sided') else None
        return plots.plot_pvals(
            output_dir=out_dir, 
            pvals_one_sided=pvals_one_sided,
            distribution=self.pval_dist,
            **kwargs
        )
    
    def plot_aberrant_per_sample(self, out_dir: str = None, **kwargs):
        """
        Plot number of aberrant proteins per sample.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = result.plot_aberrant_per_sample()  # Interactive use
            >>> plot.draw()
            >>> result.plot_aberrant_per_sample(out_dir='output/')  # Save plot
        """
        from . import plots
        # Build protrider_summary DataFrame on the fly
        ae_out = self.df_out
        ae_in = self.dataset.data
        raw_in = self.dataset.raw_data
        zscores = self.df_Z
        pvals = self.df_pvals
        pvals_adj = self.df_pvals_adj
        log2fc = self.log2fc
        fc = self.fc
        presence = self.df_presence

        ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_EXPECTED_LOG2INT'}))
        ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2INT'}))
        raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_INT'}))
        zscores = (zscores.reset_index().melt(id_vars='sampleID')
                   .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_ZSCORE'}))
        pvals = (pvals.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PVALUE'}))
        pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PADJ'}))
        log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2FC'}))
        fc = (fc.reset_index().melt(id_vars='sampleID')
              .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_FC'}))

        merge_cols = ['sampleID', 'proteinID']
        protrider_summary = (ae_in.merge(ae_out, on=merge_cols)
                  .merge(raw_in, on=merge_cols)
                  .merge(zscores, on=merge_cols)
                  .merge(pvals, on=merge_cols)
                  .merge(pvals_adj, on=merge_cols)
                  .merge(log2fc, on=merge_cols)
                  .merge(fc, on=merge_cols)
                  ).reset_index(drop=True)

        if presence is not None:
            presence = (presence.reset_index().melt(id_vars='sampleID')
                        .rename(columns={'variable': 'proteinID', 'value': 'pred_presence_probability'}))
            protrider_summary = protrider_summary.merge(presence, on=merge_cols).reset_index(drop=True)

        protrider_summary['PROTEIN_outlier'] = protrider_summary['PROTEIN_PADJ'].apply(
            lambda x: x <= self.outlier_threshold)
        protrider_summary['pvalDistribution'] = self.pval_dist
        
        return plots.plot_aberrant_per_sample(
            output_dir=out_dir,
            protrider_summary=protrider_summary,
            **kwargs
        )
    
    def plot_expected_vs_observed(self, protein_id: str, out_dir: str = None, **kwargs):
        """
        Plot expected vs observed intensities for a specific protein.
        
        Args:
            protein_id: Protein identifier to plot
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = result.plot_expected_vs_observed('PROT123')  # Interactive use
            >>> plot.draw()
            >>> result.plot_expected_vs_observed('PROT123', out_dir='output/')  # Save plot
        """
        from . import plots
        # Prepare data for plotting - transpose to match expected format
        processed_input = self.dataset.data.T.reset_index()
        processed_input.columns.name = None
        processed_input = processed_input.rename(columns={'sampleID': 'proteinID'})
        
        output_data = self.df_out.T.reset_index()
        output_data.columns.name = None
        output_data = output_data.rename(columns={'sampleID': 'proteinID'})
        
        # Build protrider_summary on the fly (same as plot_aberrant_per_sample)
        ae_out = self.df_out
        ae_in = self.dataset.data
        raw_in = self.dataset.raw_data
        zscores = self.df_Z
        pvals = self.df_pvals
        pvals_adj = self.df_pvals_adj
        log2fc = self.log2fc
        fc = self.fc
        presence = self.df_presence

        ae_out = (ae_out.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_EXPECTED_LOG2INT'}))
        ae_in = (ae_in.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2INT'}))
        raw_in = (raw_in.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_INT'}))
        zscores = (zscores.reset_index().melt(id_vars='sampleID')
                   .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_ZSCORE'}))
        pvals = (pvals.reset_index().melt(id_vars='sampleID')
                 .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PVALUE'}))
        pvals_adj = (pvals_adj.reset_index().melt(id_vars='sampleID')
                     .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_PADJ'}))
        log2fc = (log2fc.reset_index().melt(id_vars='sampleID')
                  .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_LOG2FC'}))
        fc = (fc.reset_index().melt(id_vars='sampleID')
              .rename(columns={'variable': 'proteinID', 'value': 'PROTEIN_FC'}))

        merge_cols = ['sampleID', 'proteinID']
        protrider_summary = (ae_in.merge(ae_out, on=merge_cols)
                  .merge(raw_in, on=merge_cols)
                  .merge(zscores, on=merge_cols)
                  .merge(pvals, on=merge_cols)
                  .merge(pvals_adj, on=merge_cols)
                  .merge(log2fc, on=merge_cols)
                  .merge(fc, on=merge_cols)
                  ).reset_index(drop=True)

        if presence is not None:
            presence = (presence.reset_index().melt(id_vars='sampleID')
                        .rename(columns={'variable': 'proteinID', 'value': 'pred_presence_probability'}))
            protrider_summary = protrider_summary.merge(presence, on=merge_cols).reset_index(drop=True)

        protrider_summary['PROTEIN_outlier'] = protrider_summary['PROTEIN_PADJ'].apply(
            lambda x: x <= self.outlier_threshold)
        protrider_summary['pvalDistribution'] = self.pval_dist
        
        return plots.plot_expected_vs_observed(
            protein_id=protein_id,
            output_dir=out_dir,
            processed_input=processed_input,
            output_data=output_data,
            protrider_summary=protrider_summary,
            **kwargs
        )


def run(config: ProtriderConfig) -> Tuple[Result, ModelInfo]:
    """
    Run PROTRIDER protein outlier detection.
    
    Automatically dispatches to cross-validation or standard mode based on config.cross_val.
    All inputs including data files are specified in the config.
    
    Args:
        config: ProtriderConfig object with all configuration parameters including:
                - input_intensities: File path (str)
                  * File format: columns = samples, rows = proteins
                - sample_annotation: File path (str) or None
                  * Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo)
        - Result: Contains all output dataframes (residuals, p-values, z-scores, etc.)
        - ModelInfo: Contains model metadata (q, learning_rate, losses, etc.)
                    For CV runs, includes df_folds with fold assignments
    
    Examples:
        >>> # Using file paths (file format: columns = samples, rows = proteins)
        >>> config = ProtriderConfig(
        ...     out_dir='output',
        ...     input_intensities='data.csv',  # Columns = samples
        ...     sample_annotation='annotations.csv',
        ...     cross_val=False
        ... )
        >>> result, model_info = run(config)
        
        >>> # With cross-validation
        >>> config_cv = ProtriderConfig(
        ...     out_dir='output',
        ...     input_intensities='data.csv',
        ...     cross_val=True,
        ...     n_folds=5
        ... )
        >>> result, model_info = run(config_cv)
    """
    if config.cross_val:
        logger.info('Running PROTRIDER with cross-validation')
        return _run_protrider_cv(config, config.input_intensities, config.sample_annotation)
    else:
        logger.info('Running PROTRIDER in standard mode')
        return _run_protrider_standard(config, config.input_intensities, config.sample_annotation)


def _run_protrider_standard(
    config: ProtriderConfig,
    input_intensities: Union[str, pd.DataFrame],
    sample_annotation: Union[str, pd.DataFrame, None]
) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection in a single run (internal function).
    
    Args:
        config: ProtriderConfig object with all configuration parameters
        input_intensities: Protein intensities as file path or pandas DataFrame
                          - File: columns = samples, rows = proteins
                          - DataFrame: rows = samples, columns = proteins
        sample_annotation: Sample annotations as file path, DataFrame, or None
                          - Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo)
    """
    # 1. Initialize dataset
    logger.info('Initializing dataset')
    dataset = ProtriderDataset(input_intensities=input_intensities,
                               index_col=config.index_col,
                               sa_file=sample_annotation,
                               cov_used=config.cov_used,
                               log_func=config.log_func,
                               maxNA_filter=config.max_allowed_NAs_per_protein,
                               device=config.device_torch)

    # 2. Find latent dim
    logger.info('Finding latent dimension')
    q = find_latent_dim(dataset, method=config.find_q_method,
                        # Params for grid search method
                        inj_freq=config.inj_freq,
                        inj_mean=config.inj_mean,
                        inj_sd=config.inj_sd,
                        init_wPCA=config.init_pca,
                        n_layers=config.n_layers,
                        h_dim=config.h_dim,
                        n_epochs=config.gs_epochs if config.gs_epochs else config.n_epochs,
                        learning_rate=config.lr,
                        batch_size=config.batch_size,
                        pval_sided=config.pval_sided,
                        pval_dist=config.pval_dist,
                        out_dir=config.out_dir,
                        device=config.device_torch,
                        presence_absence=config.presence_absence,
                        lambda_bce=config.lambda_presence_absence,
                        n_jobs=config.n_jobs
                        )

    logger.info(
        f'Latent dimension found with method {config.find_q_method}: {q}')

    # 3. Init model with found latent dim
    model = init_model(dataset, q,
                       init_wPCA=config.init_pca,
                       n_layer=config.n_layers,
                       h_dim=config.h_dim,
                       device=config.device_torch,
                       presence_absence=config.presence_absence if config.n_layers == 1 else False
                       )
    criterion = MSEBCELoss(
        presence_absence=config.presence_absence, lambda_bce=config.lambda_presence_absence)
    logger.info('Model:\n%s', model)
    logger.info('Device: %s', config.device_torch)

    # 4. Compute initial loss
    df_out, df_presence, init_loss, init_mse_loss, init_bce_loss = _inference(
        dataset, model, criterion)
    logger.info('Initial loss after model init: %s, mse loss: %s, bce loss: %s', init_loss, init_mse_loss,
                init_bce_loss)
    final_loss = 10**4
    train_losses = []
    if config.autoencoder_training:
        logger.info('Fitting model')
        # 5. Train model
        _, _, _, train_losses = train(dataset, model, criterion, n_epochs=config.n_epochs, learning_rate=float(config.lr),
                                      batch_size=config.batch_size)
        df_out, df_presence, final_loss, final_mse_loss, final_bce_loss = _inference(
            dataset, model, criterion)
        logger.info('Final loss: %s, mse loss: %s, bce loss: %s',
                    final_loss, final_mse_loss, final_bce_loss)
    else:
        final_loss = init_loss

    # 6. Compute residuals, pvals, zscores
    logger.info('Computing statistics')
    df_res = dataset.data - df_out  # log data - pred data

    mu, sigma, df0 = fit_residuals(df_res.values, dis=config.pval_dist, n_jobs=config.n_jobs)
    pvals, Z = get_pvals(df_res.values,
                         mu=mu,
                         sigma=sigma,
                         df0=df0,
                         how=config.pval_sided,
                         dis=config.pval_dist, n_jobs=config.n_jobs)
    pvals_one_sided, _ = get_pvals(df_res.values,
                                   mu=mu,
                                   sigma=sigma,
                                   df0=df0,
                                   how='left',
                                   dis=config.pval_dist, n_jobs=config.n_jobs)

    pvals_adj = adjust_pvals(pvals, method=config.pval_adj)
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config.pseudocount, outlier_threshold=config.outlier_threshold,
                             base_fn=config.base_fn, pval_dist=config.pval_dist)
    model_info = ModelInfo(q=np.array(q), learning_rate=np.array(config.lr),
                           n_epochs=np.array(config.n_epochs), test_loss=np.array(final_loss),
                           train_losses=np.array(train_losses), df_folds=None)
    return result, model_info


def _run_protrider_cv(
    config: ProtriderConfig,
    input_intensities: Union[str, pd.DataFrame],
    sample_annotation: Union[str, pd.DataFrame, None]
) -> Tuple[Result, ModelInfo]:
    """
    Perform protein outlier detection with cross-validation (internal function).
    
    Args:
        config: ProtriderConfig object with all configuration parameters
        input_intensities: Protein intensities as file path or pandas DataFrame
                          - File: columns = samples, rows = proteins
                          - DataFrame: rows = samples, columns = proteins
        sample_annotation: Sample annotations as file path, DataFrame, or None
                          - Format: rows = samples

    Returns:
        Tuple of (Result, ModelInfo) - ModelInfo.df_folds contains fold assignments for CV
    """
    # If fit_every_fold is set to True, the model will estimate the residual distribution parameters on every fold
    # using the train-val set.
    # If set to False, the model will estimate the residual distribution parameters on the final test residuals
    fit_every_fold = config.fit_every_fold

    # 1. Initialize cross validation generator
    logger.info('Initializing cross validation')
    if config.n_folds is not None:
        cv_gen = ProtriderKfoldCVGenerator(input_intensities, sample_annotation, config.index_col,
                                           config.cov_used, config.max_allowed_NAs_per_protein, config.log_func,
                                           num_folds=config.n_folds, device=config.device_torch)
    else:
        cv_gen = ProtriderLOOCVGenerator(input_intensities, sample_annotation, config.index_col, config.cov_used,
                                         config.max_allowed_NAs_per_protein, config.log_func, device=config.device_torch)
    dataset = cv_gen.dataset
    criterion = MSEBCELoss(
        presence_absence=config.presence_absence, lambda_bce=config.lambda_presence_absence)
    # test results
    pvals_list = []
    Z_list = []
    df_out_list = []
    df_res_list = []
    df_presence_list = []
    test_loss_list = []
    train_losses_list = []
    q_list = []
    df0_list = []
    folds_list = []
    # 2. Loop over folds
    for fold, (train_subset, val_subset, test_subset) in enumerate(cv_gen):

        logger.info(f'Fold {fold}')
        logger.info(f'Train subset size: {len(train_subset)}')
        logger.info(f'Validation subset size: {len(val_subset)}')
        logger.info(f'Test subset size: {len(test_subset)}')

        # 3. Find latent dim
        logger.info('Finding latent dimension')
        pca_subset = ProtriderSubset.concat([train_subset, val_subset])
        q = find_latent_dim(pca_subset, method=config.find_q_method,
                            # Params for grid search method
                            inj_freq=config.inj_freq,
                            inj_mean=config.inj_mean,
                            inj_sd=config.inj_sd,
                            init_wPCA=config.init_pca,
                            n_layers=config.n_layers,
                            h_dim=config.h_dim,
                            n_epochs=config.gs_epochs if config.gs_epochs else config.n_epochs,
                            learning_rate=config.lr,
                            batch_size=config.batch_size,
                            pval_sided=config.pval_sided,
                            pval_dist=config.pval_dist,
                            out_dir=config.out_dir,
                            device=config.device_torch,
                            presence_absence=config.presence_absence,
                            lambda_bce=config.lambda_presence_absence,
                            n_jobs=config.n_jobs
                            )
        logger.info(
            f'Latent dimension found with method {config.find_q_method}: {q}')

        # 4. Init model with found latent dim
        model = init_model(train_subset, q, init_wPCA=config.init_pca, n_layer=config.n_layers,
                           h_dim=config.h_dim, device=config.device_torch, presence_absence=config.presence_absence)

        logger.info('Model:\n%s', model)
        logger.info('Device: %s', config.device_torch)

        # 5. Compute initial MSE loss
        df_out_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(
            val_subset, model, criterion)
        logger.info(f'Train loss after model init: {train_loss}')
        logger.info(f'Validation loss after model init: {val_loss}')
        if config.autoencoder_training:
            logger.info('Fitting model')
            # 6. Train model
            # todo train validate (hyperparameter tuning)
            # todo pass validation set as well
            train_losses, val_losses = train_val(train_subset, val_subset, model, criterion,
                                                 n_epochs=config.n_epochs,
                                                 learning_rate=float(config.lr),
                                                 batch_size=config.batch_size,
                                                 patience=config.early_stopping_patience,
                                                 min_delta=config.early_stopping_min_delta)
            plot_cv_loss(train_losses, val_losses, fold, config.out_dir)
            train_losses_list.append(train_losses)
        else:
            train_losses_list.append([])

        df_out_train, df_presence_train, train_loss, train_mse_loss, train_bce_loss = _inference(train_subset, model,
                                                                                                 criterion)
        df_out_val, df_presence_val, val_loss, val_mse_loss, val_bce_loss = _inference(
            val_subset, model, criterion)
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
            logger.info(
                'Estimating residual distribution parameters on train-val set')
            df_res_val = val_subset.data - df_out_val  # log data - pred data
            df_res_train = train_subset.data - df_out_train  # log data - pred data
            mu, sigma, df0 = fit_residuals(
                pd.concat([df_res_train, df_res_val]).values, dis=config.pval_dist, n_jobs=config.n_jobs)
            pvals, Z = get_pvals(df_res_test.values, mu=mu,
                                 sigma=sigma, df0=df0, how=config.pval_sided, n_jobs=config.n_jobs)
            pvals_list.append(pvals)
            Z_list.append(Z)
            df0_list.append(df0)

    df_res = pd.concat(df_res_list)
    df_out = pd.concat(df_out_list)
    df_presence = pd.concat(
        df_presence_list) if df_presence_list != [] else None
    df_folds = pd.DataFrame({'fold': folds_list, }, index=df_out.index)

    # Compute p-values on test set
    if fit_every_fold:
        pvals = np.concatenate(pvals_list)
        Z = np.concatenate(Z_list)
    else:
        logger.info('Estimating residual distribution parameters')
        mu, sigma, df0 = fit_residuals(df_res.values, dis=config.pval_dist, n_jobs=config.n_jobs)
        pvals, Z = get_pvals(df_res.values, mu=mu, sigma=sigma, df0=df0,
                             how=config.pval_sided, n_jobs=config.n_jobs)
        # Repeat df0 for each sample in the output
        df0_list = [df0] * len(df_out)

    # Compute one-sided p-values (used for some plots)
    pvals_one_sided, _ = get_pvals(df_res.values,
                                   mu=mu,
                                   sigma=sigma,
                                   df0=df0 if config.pval_dist == 't' else None,
                                   how='left',
                                   dis=config.pval_dist,
                                   n_jobs=config.n_jobs)

    pvals_adj = adjust_pvals(pvals, method=config.pval_adj)
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config.pseudocount,
                             outlier_threshold=config.outlier_threshold, base_fn=config.base_fn, pval_dist=config.pval_dist)
    model_info = ModelInfo(q=np.array(q_list), learning_rate=np.array(config.lr),
                           n_epochs=np.array(config.n_epochs), test_loss=np.array(test_loss_list),
                           train_losses=np.array(train_losses_list, dtype=object), 
                           df0=np.array(df0_list), df_folds=df_folds)
    return result, model_info


def _inference(dataset: Union[ProtriderDataset, ProtriderSubset], model: ProtriderAutoencoder, criterion: MSEBCELoss):
    X_out = model(dataset.X, dataset.torch_mask, cond=dataset.covariates)

    loss, mse_loss, bce_loss = criterion(
        X_out, dataset.X, dataset.torch_mask, detached=True)
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


def _format_results(df_out, df_res, df_presence, pvals, Z, pvals_one_sided, pvals_adj, dataset, pseudocount, outlier_threshold, base_fn, pval_dist):
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
    log2fc = np.log2(base_fn(dataset.data) + pseudocount) - \
        np.log2(base_fn(df_out) + pseudocount)
    fc = (base_fn(dataset.data) + pseudocount) / \
        (base_fn(df_out) + pseudocount)

    outs_per_sample = np.sum(df_pvals_adj.values <= outlier_threshold, axis=1)
    n_out_median = np.nanmedian(outs_per_sample)
    n_out_max = np.nanmax(outs_per_sample)
    n_out_total = np.nansum(outs_per_sample)
    logger.info(
        f'Finished computing pvalues. No. outliers per sample in median: {n_out_median}')

    logger.info(
        f'Finished computing pvalues. No. outliers per sample in median: {np.nanmedian(outs_per_sample)}')

    return Result(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence, df_pvals=df_pvals, df_Z=df_Z,
                  df_pvals_one_sided=df_pvals_one_sided, df_pvals_adj=df_pvals_adj, log2fc=log2fc, fc=fc, n_out_median=n_out_median, n_out_max=n_out_max,
                  n_out_total=n_out_total, pval_dist=pval_dist, outlier_threshold=outlier_threshold)
