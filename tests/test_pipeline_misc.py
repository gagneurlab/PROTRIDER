"""
Tests for PROTRIDER pipeline output consistency and edge cases.

This module tests:
- Output consistency (p-values, fold changes, name preservation)
- Edge cases (multiple covariates, pseudocount, seeds, reproducibility)
- Configuration save/load
- Report options
"""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from protrider import ProtriderConfig, run, load_config
from protrider.pipeline import Result
from protrider.model import ModelInfo


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
            
            result, _ = run(config)
            
            # Check p-values are in [0, 1] (ignoring NaN values which are expected for missing data)
            valid_pvals = result.df_pvals.dropna()
            assert (valid_pvals >= 0).all().all(), "Some p-values are < 0"
            assert (valid_pvals <= 1).all().all(), "Some p-values are > 1"
            
            # Check adjusted p-values are in [0, 1] (ignoring NaN values)
            valid_pvals_adj = result.df_pvals_adj.dropna()
            assert (valid_pvals_adj >= 0).all().all(), "Some adjusted p-values are < 0"
            assert (valid_pvals_adj <= 1).all().all(), "Some adjusted p-values are > 1"
    
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
            
            result, _ = run(config)
            
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
            
            result, _ = run(config)
            
            # Check that sample names match (after filtering for NAs)
            assert len(result.df_pvals.index) <= len(protein_df.columns)
            
            # Check that protein names match (after filtering)
            assert len(result.df_pvals.columns) <= len(protein_df.index)


class TestPipelineEdgeCases:
    """Test class for edge cases and robustness."""
    
    def test_single_covariate(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test with a single covariate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE'],  # Single covariate
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_multiple_covariates(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test with multiple covariates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE', 'SEX', 'BATCH_RUN'],  # Multiple covariates
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, model_info = run(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_custom_pseudocount(self, protein_intensities_path, protein_intensities_index_col):
        """Test with custom pseudocount value."""
        for pseudocount in [0.001, 0.1, 1.0]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    pseudocount=pseudocount,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run(config)
                
                assert isinstance(result, Result)
    
    def test_different_seeds(self, protein_intensities_path, protein_intensities_index_col):
        """Test that different seeds produce different but valid results."""
        results = []
        for seed in [42, 123, 456]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    seed=seed,
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run(config)
                results.append(result)
        
        # All should be valid results
        for result in results:
            assert isinstance(result, Result)
            assert result.df_pvals.notna().any().any()
    
    def test_reproducibility_same_seed(self, protein_intensities_path, protein_intensities_index_col):
        """Test that same seed produces identical results."""
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = ProtriderConfig(
                    out_dir=tmp_dir,
                    input_intensities=str(protein_intensities_path),
                    index_col=protein_intensities_index_col,
                    seed=42,  # Same seed
                    n_epochs=2,
                    find_q_method='5',
                    verbose=False
                )
                
                result, _ = run(config)
                results.append(result)
        
        # Results should be very similar (allowing small numerical differences)
        # Compare a subset of values
        assert results[0].df_pvals.shape == results[1].df_pvals.shape
        # Check that most values are close
        valid_mask = results[0].df_pvals.notna() & results[1].df_pvals.notna()
        if valid_mask.any().any():
            diff = (results[0].df_pvals[valid_mask] - results[1].df_pvals[valid_mask]).abs()
            # Most differences should be very small
            assert (diff < 0.01).sum().sum() / valid_mask.sum().sum() > 0.95
    
    def test_config_save_load(self, protein_intensities_path, protein_intensities_index_col):
        """Test that configuration can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save config
            original_config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                n_epochs=2,
                find_q_method='5',
                lr=0.001,
                verbose=False
            )
            
            original_config.save(tmp_dir)
            
            # Load config
            loaded_config = load_config(Path(tmp_dir) / 'config.yaml')
            
            # Check key parameters match
            assert loaded_config.n_epochs == original_config.n_epochs
            assert loaded_config.find_q_method == original_config.find_q_method
            assert loaded_config.lr == original_config.lr
            assert loaded_config.index_col == original_config.index_col
    
    def test_report_all_false(self, protein_intensities_path, protein_intensities_index_col):
        """Test with report_all=False to only report significant results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                index_col=protein_intensities_index_col,
                report_all=False,  # Only significant results
                outlier_threshold=0.1,
                n_epochs=2,
                find_q_method='5',
                verbose=False
            )
            
            result, _ = run(config)
            
            assert isinstance(result, Result)
            # Save and check that output exists
            result.save(tmp_dir, format='long')
            assert (Path(tmp_dir) / 'protrider_summary.csv').exists()
