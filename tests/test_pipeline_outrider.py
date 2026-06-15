"""
Tests for OUTRIDER pipeline

This module tests basic pipeline execution with different input formats
and output saving options.
"""

import pandas as pd
import tempfile
from pathlib import Path

from protrider import ProtriderConfig, run
from protrider.pipeline import Result
from protrider.model import ModelInfo


class TestPipelineOUTRIDER:
    """Test class for standard (non-CV) pipeline execution."""

    def test_run_with_file_paths(self, gene_expression_path, gene_annotation_path):
        """Test running PROTRIDER with file paths in config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ProtriderConfig(
                out_dir=tmp_dir,
                analysis='outrider',
                autoencoder_loss='NLL',
                pval_dist='nb',
                input_intensities=gene_expression_path,
                index_col='geneID',
                n_epochs=2,  # Short for testing
                gs_epochs=2,  # Short for testing
                find_q_method='5',  # Fixed q for speed
                verbose=False
            )

            result, model_info, *_ = run(config)

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
