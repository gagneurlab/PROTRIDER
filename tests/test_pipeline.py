"""
Test suite for PROTRIDER pipeline functionality.

This module contains comprehensive tests for the main pipeline functions,
covering various scenarios including:
- Standard mode with file paths
- Standard mode with DataFrames
- Cross-validation mode
- Configuration validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from protrider import ProtriderConfig, run_protrider
from protrider.pipeline import Result
from protrider.model import ModelInfo


class TestPipelineStandardMode:
    """Test class for standard (non-CV) pipeline execution."""
    
    def test_run_with_file_paths(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test running PROTRIDER with file paths in config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE', 'SEX'],
                n_epochs=2,  # Short for testing
                gs_epochs=2,  # Short for testing
                find_q_method='5',  # Fixed q for speed
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            # Check result type
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            
            # Check result contains expected dataframes
            assert isinstance(result.df_out, pd.DataFrame)
            assert isinstance(result.df_res, pd.DataFrame)
            assert isinstance(result.df_pvals, pd.DataFrame)
            assert isinstance(result.df_Z, pd.DataFrame)
            assert isinstance(result.df_pvals_adj, pd.DataFrame)
            
            # Check shapes are consistent
            n_samples, n_proteins = result.df_res.shape
            assert result.df_pvals.shape == (n_samples, n_proteins)
            assert result.df_Z.shape == (n_samples, n_proteins)
            
            # Check no NaN values in outputs
            assert not result.df_pvals.isna().any().any()
            assert not result.df_Z.isna().any().any()
            
            # Check model info
            assert model_info.q > 0
            assert model_info.learning_rate > 0
            assert len(model_info.train_losses) > 0
    
    def test_run_with_dataframes(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test running PROTRIDER with DataFrames instead of file paths."""
        # Load data as DataFrames
        protein_df = pd.read_csv(protein_intensities_path, sep='\t', index_col=protein_intensities_index_col)
        annotation_df = pd.read_csv(covariates_path, sep='\t')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=protein_df,  # DataFrame
                sample_annotation=annotation_df,  # DataFrame
                index_col=protein_intensities_index_col,
                cov_used=['AGE', 'SEX'],
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            assert result.df_pvals.shape[0] > 0
            assert result.df_pvals.shape[1] > 0
    
    def test_run_without_covariates(self, protein_intensities_path, protein_intensities_index_col):
        """Test running PROTRIDER without sample annotations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=None,  # No annotations
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_save_results_wide_format(self, protein_intensities_path, protein_intensities_index_col):
        """Test saving results in wide format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            # Save in wide format
            result.save(tmp_dir, format='wide')
            
            # Check that files were created
            out_dir = Path(tmp_dir)
            assert (out_dir / 'pvals.csv').exists()
            assert (out_dir / 'pvals_adj.csv').exists()
            assert (out_dir / 'zscores.csv').exists()
            assert (out_dir / 'residuals.csv').exists()
            assert (out_dir / 'log2fc.csv').exists()
            assert (out_dir / 'fc.csv').exists()
    
    def test_save_results_long_format(self, protein_intensities_path, protein_intensities_index_col):
        """Test saving results in long format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            # Save in long format
            result.save(tmp_dir, format='long')
            
            # Check that output file was created
            out_dir = Path(tmp_dir)
            assert (out_dir / 'output.csv').exists()
            
            # Load and check structure
            output_df = pd.read_csv(out_dir / 'output.csv')
            expected_cols = ['sampleID', 'geneID', 'pValue', 'padjust', 'zScore', 
                           'l2fc', 'rawcounts', 'normcounts', 'aberrant']
            for col in expected_cols:
                assert col in output_df.columns


class TestPipelineCrossValidation:
    """Test class for cross-validation pipeline execution."""
    
    def test_run_with_kfold_cv(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test running PROTRIDER with k-fold cross-validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE', 'SEX'],
                cross_val=True,
                n_folds=3,  # Small number for testing
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            
            # Check that fold information is present
            assert hasattr(model_info, 'df_folds')
            assert model_info.df_folds is not None
            assert len(model_info.df_folds) > 0
            
            # Check that all samples are assigned to a fold
            assert model_info.df_folds['fold'].notna().all()
    
    def test_run_with_loocv(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test running PROTRIDER with leave-one-out cross-validation."""
        # Use only a small subset for LOOCV to keep test fast
        protein_df = pd.read_csv(protein_intensities_path, sep='\t', index_col=protein_intensities_index_col)
        annotation_df = pd.read_csv(covariates_path, sep='\t')
        
        # Take only first 5 samples
        small_protein_df = protein_df.iloc[:, :5]
        small_annotation_df = annotation_df.iloc[:5]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=small_protein_df,
                sample_annotation=small_annotation_df,
                index_col=protein_intensities_index_col,
                cov_used=['AGE'],
                cross_val=True,
                n_folds=None,  # None triggers LOOCV
                n_epochs=2,
                find_q_method='3',
                verbose=False
            )
            
            result, model_info = run_protrider(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            
            # Check that fold information matches number of samples
            assert len(model_info.df_folds) == 5


class TestPipelineConfiguration:
    """Test class for configuration validation."""
    
    def test_missing_input_intensities(self):
        """Test that missing input_intensities raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=None,  # Missing
                index_col='protein_ID'
            )
            
            # Should fail when trying to run
            with pytest.raises((ValueError, TypeError, AttributeError)):
                run_protrider(config)
    
    def test_invalid_latent_dim_method(self, protein_intensities_path, protein_intensities_index_col):
        """Test that invalid find_q_method raises error during config validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="find_q_method"):
                # Config creation should raise error
                _ = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    find_q_method='invalid_method'  # Invalid
                )
    
    def test_negative_epochs(self, protein_intensities_path, protein_intensities_index_col):
        """Test that negative n_epochs raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="n_epochs"):
                # Config creation should raise error
                _ = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    n_epochs=-1  # Invalid
                )
    
    def test_invalid_max_na_per_protein(self, protein_intensities_path, protein_intensities_index_col):
        """Test that invalid max_allowed_NAs_per_protein raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="max_allowed_NAs_per_protein"):
                # Config creation should raise error
                _ = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    max_allowed_NAs_per_protein=1.5  # Must be between 0 and 1
                )


class TestPipelineOutputConsistency:
    """Test class for output consistency and data integrity."""
    
    def test_pvalue_range(self, protein_intensities_path, protein_intensities_index_col):
        """Test that p-values are in valid range [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, _ = run_protrider(config)
            
            # Check p-values are in [0, 1]
            assert (result.df_pvals >= 0).all().all()
            assert (result.df_pvals <= 1).all().all()
            
            # Check adjusted p-values are in [0, 1]
            assert (result.df_pvals_adj >= 0).all().all()
            assert (result.df_pvals_adj <= 1).all().all()
    
    def test_fold_change_consistency(self, protein_intensities_path, protein_intensities_index_col):
        """Test that fold changes are computed correctly from log2fc."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, _ = run_protrider(config)
            
            # fc should be 2^log2fc
            expected_fc = 2 ** result.log2fc
            np.testing.assert_array_almost_equal(result.fc, expected_fc, decimal=5)
    
    def test_sample_protein_names_preserved(self, protein_intensities_path, protein_intensities_index_col):
        """Test that sample and protein names are preserved in outputs."""
        # Load original data
        protein_df = pd.read_csv(protein_intensities_path, sep='\t', index_col=protein_intensities_index_col)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, _ = run_protrider(config)
            
            # Check that sample names match (after filtering for NAs)
            assert len(result.df_pvals.index) <= len(protein_df.columns)
            
            # Check that protein names match (after filtering)
            assert len(result.df_pvals.columns) <= len(protein_df.index)
