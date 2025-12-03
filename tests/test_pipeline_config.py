"""
Tests for PROTRIDER configuration validation.

This module tests that invalid configurations are properly rejected.
"""

import pytest
import tempfile

from protrider import ProtriderConfig, run


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
                run(config)
    
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
