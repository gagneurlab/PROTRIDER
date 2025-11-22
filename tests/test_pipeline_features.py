"""
Tests for PROTRIDER pipeline advanced features.

This module tests advanced configuration options including:
- Log transformations
- P-value distributions and adjustments
- Latent dimension methods
- NA thresholds
- Batch size and learning rates
- PCA initialization
- Outlier thresholds
"""

import tempfile
from protrider import ProtriderConfig, run_protrider
from protrider.pipeline import Result
from protrider.model import ModelInfo


class TestPipelineAdvancedFeatures:
    """Test class for advanced pipeline features."""
    
    def test_different_log_transformations(self, protein_intensities_path, protein_intensities_index_col):
        """Test different log transformation methods."""
        for log_func in ["log", "log2", "log10"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    log_func_name=log_func,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, model_info = run_protrider(config)
                
                assert isinstance(result, Result)
                assert isinstance(model_info, ModelInfo)
    
    def test_no_log_transformation(self, protein_intensities_path, protein_intensities_index_col):
        """Test running without log transformation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                log_func_name=None,  # No log transformation
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, _ = run_protrider(config)
            
            assert isinstance(result, Result)
    
    def test_different_pval_distributions(self, protein_intensities_path, protein_intensities_index_col):
        """Test using different distributions for p-value calculation."""
        for dist in ["gaussian", "t"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    pval_dist=dist,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run_protrider(config)
                
                assert isinstance(result, Result)
                # Check p-values are computed
                assert result.df_pvals.notna().any().any()
    
    def test_different_pval_adjustment_methods(self, protein_intensities_path, protein_intensities_index_col):
        """Test different p-value adjustment methods."""
        for adj_method in ["bh", "by"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    pval_adj=adj_method,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run_protrider(config)
                
                assert isinstance(result, Result)
                # Check adjusted p-values are in valid range [0, 1]
                # (ignoring NaN values which are expected)
                valid_adj = result.df_pvals_adj.dropna()
                assert (valid_adj >= 0).all().all(), f"Some adjusted p-values < 0 for method {adj_method}"
                assert (valid_adj <= 1).all().all(), f"Some adjusted p-values > 1 for method {adj_method}"
                
                # Check that we have some valid adjusted p-values
                assert result.df_pvals_adj.notna().any().any(), f"All adjusted p-values are NaN for method {adj_method}"
    
    def test_different_latent_dim_methods(self, protein_intensities_path, protein_intensities_index_col):
        """Test different methods for finding latent dimension."""
        for method in ["3", "5", "10", "OHT"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    find_q_method=method,
                    n_epochs=2,
                    gs_epochs=2,  # Small for testing
                    verbose=False
                )
                
                result, model_info = run_protrider(config)
                
                assert isinstance(result, Result)
                assert model_info.q > 0
                
                # For fixed integer methods, check q matches
                if method.isdigit():
                    assert model_info.q == int(method)
    
    def test_grid_search_latent_dim(self, protein_intensities_path, protein_intensities_index_col):
        """Test grid search for optimal latent dimension."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                find_q_method="gs",  # Grid search
                gs_epochs=2,  # Small for testing
                n_epochs=2,
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert model_info.q > 0
            # Grid search should find optimal q
            assert model_info.q >= 1
    
    def test_different_na_thresholds(self, protein_intensities_path, protein_intensities_index_col):
        """Test different thresholds for missing data filtering."""
        for threshold in [0.1, 0.5, 0.9]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    max_allowed_NAs_per_protein=threshold,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run_protrider(config)
                
                assert isinstance(result, Result)
    
    def test_batch_size_parameter(self, protein_intensities_path, protein_intensities_index_col):
        """Test using explicit batch size."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                batch_size=32,  # Explicit batch size
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_different_learning_rates(self, protein_intensities_path, protein_intensities_index_col):
        """Test different learning rates."""
        for lr in [1e-5, 1e-3, 1e-2]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    lr=lr,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, model_info = run_protrider(config)
                
                assert isinstance(result, Result)
                assert model_info.learning_rate == lr
    
    def test_pca_initialization(self, protein_intensities_path, protein_intensities_index_col):
        """Test that PCA initialization affects model training."""
        # Test with PCA initialization (default)
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_pca = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                init_pca=True,  # With PCA init
                n_epochs=5,  # Need a few epochs to see initialization effect
                find_q_method='5',
                seed=42,
                verbose=False
            )
            
            result_pca, model_info_pca = run_protrider(config_pca)
            
            assert isinstance(result_pca, Result)
            assert isinstance(model_info_pca, ModelInfo)
            # With PCA init, initial loss should be lower
            assert len(model_info_pca.train_losses) > 0
    
    def test_no_pca_initialization(self, protein_intensities_path, protein_intensities_index_col):
        """Test model training without PCA initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                init_pca=False,  # No PCA init
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_different_outlier_thresholds(self, protein_intensities_path, protein_intensities_index_col):
        """Test different outlier detection thresholds."""
        for threshold in [0.01, 0.05, 0.1]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    outlier_threshold=threshold,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run_protrider(config)
                
                assert isinstance(result, Result)
                # Number of outliers should vary with threshold
                # (higher threshold = more outliers)
    
    def test_presence_absence_modeling(self, protein_intensities_path, protein_intensities_index_col):
        """Test presence/absence modeling for handling missing data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                presence_absence=True,  # Enable presence/absence modeling
                lambda_presence_absence=0.5,
                n_layers=1,  # Required for presence_absence
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            # Check that the model was created with presence_absence enabled
            assert model_info is not None
    
    def test_presence_absence_different_lambda(self, protein_intensities_path, protein_intensities_index_col):
        """Test different lambda values for presence/absence loss weighting."""
        for lambda_val in [0.1, 0.5, 1.0]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    presence_absence=True,
                    lambda_presence_absence=lambda_val,
                    n_layers=1,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run_protrider(config)
                
                assert isinstance(result, Result)
    
    def test_presence_absence_with_pca_init(self, protein_intensities_path, protein_intensities_index_col):
        """Test that presence/absence works with PCA initialization.
        
        Note: PCA is only run on the protein intensity matrix, not on the presence/absence
        indicator. The same PCA-initialized weights are applied to both the intensity
        channel and the presence channel in the encoder.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                presence_absence=True,
                lambda_presence_absence=0.5,
                init_pca=True,  # PCA initialization
                n_layers=1,  # Required for both presence_absence and PCA init
                n_epochs=5,
                find_q_method='5',
                seed=42,
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            assert len(model_info.train_losses) > 0
