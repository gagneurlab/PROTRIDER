import numpy as np
import pandas as pd
from typing import Union, Tuple, Literal, Optional
import logging
import torch
from dataclasses import dataclass
from pathlib import Path

from .model import train, train_val, MSEBCELoss, ProtriderAutoencoder, find_latent_dim, init_model, ModelInfo
from .datasets import ProtriderDataset, ProtriderSubset, ProtriderKfoldCVGenerator, ProtriderLOOCVGenerator
from .stats import get_pvals, fit_residuals, adjust_pvals
from .plots import plot_cv_loss
from .config import ProtriderConfig


__all__ = ["run"]

logger = logging.getLogger(__name__)


def save_model(model: ProtriderAutoencoder, checkpoint_path: str, q: int) -> None:
    """Save model state dict and metadata to checkpoint path.
    
    Args:
        model: Trained ProtriderAutoencoder model
        checkpoint_path: Path where to save the model checkpoint
        q: Latent dimension
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'q': q,
        'n_layers': model.n_layers,
        'presence_absence': model.presence_absence,
    }, checkpoint_path)
    
    logger.info(f'Saved model to {checkpoint_path}')


def load_model(dataset: Union[ProtriderDataset, ProtriderSubset], checkpoint_path: str, 
               config: ProtriderConfig) -> Tuple[Optional[ProtriderAutoencoder], Optional[int]]:
    """Load model from checkpoint path if it exists.
    
    Args:
        dataset: Dataset used for model initialization
        checkpoint_path: Path to the model checkpoint file
        config: ProtriderConfig object
        
    Returns:
        Tuple of (model, q) if model exists and loads successfully, (None, None) otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.info(f'No existing model found at {checkpoint_path}')
        return None, None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=config.device_torch, weights_only=False)
        q = checkpoint['q']
        n_layers = checkpoint['n_layers']
        presence_absence = checkpoint.get('presence_absence', False)
        
        logger.info(f'Loading model from {checkpoint_path} (q={q}, n_layers={n_layers})')
        
        # Initialize model with saved architecture
        n_cov = dataset.covariates.shape[1]
        n_prots = dataset.X.shape[1]
        model = ProtriderAutoencoder(
            in_dim=n_prots, 
            latent_dim=q, 
            n_layers=n_layers, 
            h_dim=config.h_dim, 
            n_cov=n_cov,
            prot_means=None,
            presence_absence=presence_absence
        )
        model.double().to(config.device_torch)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info('Successfully loaded model')
        return model, q
        
    except Exception as e:
        logger.warning(f'Failed to load model from {checkpoint_path}: {e}')
        return None, None


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
    degrees_freedom: np.ndarray = None  # Degrees of freedom for t-distribution, if applicable
    pval_dist: str = 'gaussian'  # Distribution used for p-value computation
    outlier_threshold: float = 0.1  # Threshold for determining outliers
    
    def save(self, out_dir: str, format: Literal["wide", "long"] = "wide", 
             include_all: bool = False):
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
        # save degrees of freedom if applicable
        if self.degrees_freedom is not None:
            out_p = f'{out_dir}/degrees_of_freedom.csv'
            df_df = pd.DataFrame(self.degrees_freedom, index=self.dataset.data.columns, columns=['degrees_of_freedom'])
            df_df.to_csv(out_p, header=True, index=True)
            logger.info(f'Saved degrees of freedom to {out_p}')

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
            
        elif format == "long":
            logger.info('=== Saving results in long format ===')
            
            # Create a multi-index dataframe with all values
            dfs_to_melt = {
                'PROTEIN_LOG2INT': self.dataset.data,
                'PROTEIN_EXPECTED_LOG2INT': self.df_out,
                'PROTEIN_INT': self.dataset.raw_data,
                'PROTEIN_ZSCORE': self.df_Z,
                'PROTEIN_PVALUE': self.df_pvals,
                'PROTEIN_PADJ': self.df_pvals_adj,
                'PROTEIN_LOG2FC': self.log2fc,
                'PROTEIN_FC': self.fc,
            }
            
            if self.df_presence is not None:
                dfs_to_melt['pred_presence_probability'] = self.df_presence
            
            # Concatenate all dataframes along columns with a multi-index
            combined = pd.concat(dfs_to_melt, axis=1)
            
            # Melt once instead of multiple times (use future_stack=True for pandas 2.1+)
            df_res = combined.stack(future_stack=True).reset_index()
            df_res.columns = ['sampleID', 'proteinID'] + list(dfs_to_melt.keys())

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
                               device=config.device_torch,
                               input_format=config.input_format)

    # 2. Determine checkpoint path and try to load existing model
    model = None
    q = None
    # Use custom checkpoint path if specified, otherwise default to out_dir/model.pt
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
    else:
        checkpoint_path = Path(config.out_dir) / 'model.pt'
    
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f'Attempting to load model from {checkpoint_path}')
        model, q = load_model(dataset, str(checkpoint_path), config)
    
    # 3. If model not loaded, find latent dim and initialize new model
    if model is None:
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
                            common_degrees_freedom=config.common_degrees_freedom,
                            n_jobs=config.n_jobs,
                            patience=config.patience,
                            min_delta=config.min_delta
                            )

        logger.info(
            f'Latent dimension found with method {config.find_q_method}: {q}')

        # Init model with found latent dim
        model = init_model(dataset, q,
                           init_wPCA=config.init_pca,
                           n_layer=config.n_layers,
                           h_dim=config.h_dim,
                           device=config.device_torch,
                           presence_absence=config.presence_absence if config.n_layers == 1 else False
                           )
    else:
        logger.info(f'Using loaded model with q={q}')
    
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
    
    # 5. Train model if needed (skip if model was loaded from checkpoint)
    model_was_loaded = (checkpoint_path and checkpoint_path.exists() and q is not None)
    should_train = config.autoencoder_training and not model_was_loaded
    
    if should_train:
        wandb = None
        if config.use_wandb:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=config.wandb_project,
                       name=config.wandb_name,
                       config={
                           'latent_dim': q,
                           'n_layers': config.n_layers,
                           'h_dim': config.h_dim,
                           'learning_rate': config.lr,
                           'n_epochs': config.n_epochs,
                           'batch_size': config.batch_size,
                           'presence_absence': config.presence_absence,
                           'lambda_bce': config.lambda_presence_absence
                       })

        logger.info('Fitting model')
        _, _, _, train_losses = train(dataset, model, criterion, n_epochs=config.n_epochs, learning_rate=float(config.lr),
                                      batch_size=config.batch_size, wandb=wandb, patience=config.patience, min_delta=config.min_delta)
        df_out, df_presence, final_loss, final_mse_loss, final_bce_loss = _inference(
            dataset, model, criterion)
        logger.info('Final loss: %s, mse loss: %s, bce loss: %s',
                    final_loss, final_mse_loss, final_bce_loss)
        
        # Save the trained model to checkpoint
        if checkpoint_path:
            save_model(model, str(checkpoint_path), q)
            if config.use_wandb:
                wandb.log_model(str(checkpoint_path), 'protrider_model')
        
        if config.use_wandb:
            wandb.finish()
    else:
        if model_was_loaded:
            logger.info('Skipping training - using loaded model from checkpoint')
        final_loss = init_loss

    # 6. Compute residuals, pvals, zscores
    logger.info('Computing statistics')
    df_res = dataset.data - df_out  # log data - pred data

    mu, sigma, degrees_freedom = fit_residuals(df_res.values, dis=config.pval_dist, n_jobs=config.n_jobs, use_common_df=config.common_degrees_freedom)
    pvals, Z = get_pvals(df_res.values,
                         mu=mu,
                         sigma=sigma,
                         df=degrees_freedom,
                         how=config.pval_sided,
                         dis=config.pval_dist, n_jobs=config.n_jobs)
    pvals_one_sided, _ = get_pvals(df_res.values,
                                   mu=mu,
                                   sigma=sigma,
                                   df=degrees_freedom,
                                   how='left',
                                   dis=config.pval_dist, n_jobs=config.n_jobs)

    pvals_adj = adjust_pvals(pvals, method=config.pval_adj)
    result = _format_results(dataset=dataset, df_out=df_out, df_res=df_res, df_presence=df_presence,
                             pvals=pvals, Z=Z, pvals_one_sided=pvals_one_sided, pvals_adj=pvals_adj,
                             pseudocount=config.pseudocount, outlier_threshold=config.outlier_threshold,
                             base_fn=config.base_fn, pval_dist=config.pval_dist, degrees_freedom=degrees_freedom)
    model_info = ModelInfo(q=np.array(q), learning_rate=np.array(config.lr),
                           n_epochs=np.array(config.n_epochs), test_loss=np.array(final_loss),
                           train_losses=np.array(train_losses), df_folds=None)
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


def _format_results(df_out, df_res, df_presence, pvals, Z, pvals_one_sided, pvals_adj, dataset, pseudocount, outlier_threshold, base_fn, pval_dist, degrees_freedom=None):
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
                  n_out_total=n_out_total, pval_dist=pval_dist, outlier_threshold=outlier_threshold, degrees_freedom=degrees_freedom)
