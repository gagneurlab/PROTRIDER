"""
Tests for PROTRIDER standard (non-CV) pipeline mode.

This module tests basic pipeline execution with different input formats
and output saving options.
"""

import pandas as pd
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
            
            # Note: NaN p-values and Z-scores are expected where input data has missing values
            # Just check that we have some valid (non-NaN) p-values
            assert result.df_pvals.notna().any().any(), "All p-values are NaN"
            assert result.df_Z.notna().any().any(), "All Z-scores are NaN"
            
            # Check model info
            assert model_info.q > 0
            assert model_info.learning_rate > 0
            assert len(model_info.train_losses) > 0
    
    def test_run_with_alternate_file_format(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test running PROTRIDER with CSV file format."""
        # Read and save as CSV to test CSV reading
        protein_df = pd.read_csv(protein_intensities_path, sep='\t', index_col=protein_intensities_index_col)
        annotation_df = pd.read_csv(covariates_path, sep='\t')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save as CSV files
            csv_protein_path = Path(tmp_dir) / 'proteins.csv'
            csv_annotation_path = Path(tmp_dir) / 'annotations.csv'
            
            protein_df.to_csv(csv_protein_path)
            annotation_df.to_csv(csv_annotation_path, index=False)
            
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(csv_protein_path),
                sample_annotation=str(csv_annotation_path),
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
            
            # Check that output file was created (long format saves as protrider_summary.csv)
            out_dir = Path(tmp_dir)
            assert (out_dir / 'protrider_summary.csv').exists()
            
            # Load and check structure
            output_df = pd.read_csv(out_dir / 'protrider_summary.csv')
            # Check for expected columns (actual column names from PROTRIDER)
            expected_cols = ['sampleID', 'proteinID', 'PROTEIN_PVALUE', 'PROTEIN_PADJ', 
                           'PROTEIN_ZSCORE', 'PROTEIN_LOG2FC', 'PROTEIN_outlier']
            for col in expected_cols:
                assert col in output_df.columns, f"Column '{col}' not found in output"
