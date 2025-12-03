"""
Tests for PROTRIDER cross-validation pipeline mode.

This module tests k-fold and leave-one-out cross-validation modes.
"""

import pandas as pd
import tempfile
from pathlib import Path

from protrider import ProtriderConfig, run
from protrider.pipeline import Result
from protrider.model import ModelInfo


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
                seed=42,  # Set seed for reproducibility
                verbose=False
            )
            
            result, model_info = run(config)
            
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
        # Note: Need at least 10 samples for stable DF estimation (df > 2 requirement)
        protein_df = pd.read_csv(protein_intensities_path, sep='\t', index_col=protein_intensities_index_col)
        annotation_df = pd.read_csv(covariates_path, sep='\t')
        
        # Take first 10 samples and save as temporary files
        small_protein_df = protein_df.iloc[:, :10]
        small_annotation_df = annotation_df.iloc[:10]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save subset to temporary files
            small_protein_path = Path(tmp_dir) / 'small_proteins.tsv'
            small_annotation_path = Path(tmp_dir) / 'small_annotations.tsv'
            
            small_protein_df.to_csv(small_protein_path, sep='\t')
            small_annotation_df.to_csv(small_annotation_path, sep='\t', index=False)
            
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(small_protein_path),
                sample_annotation=str(small_annotation_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE'],
                cross_val=True,
                n_folds=None,  # None triggers LOOCV
                n_epochs=2,
                find_q_method='3',
                seed=42,  # Set seed for reproducibility
                verbose=False
            )
            
            result, model_info = run(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
            
            # Check that fold information matches number of samples
            assert len(model_info.df_folds) == 10
    
    def test_cv_with_early_stopping(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test cross-validation with early stopping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE'],
                cross_val=True,
                n_folds=2,
                n_epochs=100,  # High epochs to trigger early stopping
                early_stopping_patience=5,
                early_stopping_min_delta=0.001,
                find_q_method='3',
                seed=42,  # Set seed for reproducibility
                verbose=False
            )
            
            result, model_info = run(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
    
    def test_cv_fit_every_fold(self, protein_intensities_path, covariates_path, protein_intensities_index_col):
        """Test cross-validation with fit_every_fold enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                input_intensities=str(protein_intensities_path),
                sample_annotation=str(covariates_path),
                index_col=protein_intensities_index_col,
                cov_used=['AGE'],
                cross_val=True,
                n_folds=2,
                fit_every_fold=True,  # Refit hyperparameters each fold
                n_epochs=2,
                gs_epochs=2,
                find_q_method='OHT',
                seed=42,  # Set seed for reproducibility
                verbose=False
            )
            
            result, model_info = run(config)
            
            assert isinstance(result, Result)
            assert isinstance(model_info, ModelInfo)
