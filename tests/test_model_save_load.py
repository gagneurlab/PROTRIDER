"""
Test model checkpointing functionality.

Run this test with:
    pytest tests/test_model_save_load.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch

from protrider import ProtriderConfig, run
from protrider.pipeline import save_model, load_model
from protrider.model import ProtriderAutoencoder
from protrider.datasets import ProtriderDataset


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Load sample dataset for testing."""
    return ProtriderDataset(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        sa_file='sample_data/sample_annotations.tsv',
        index_col='protein_ID',
        log_func=None,
        maxNA_filter=0.3,
        device=torch.device('cpu'),
        input_format='proteins_as_rows'
    )


def test_save_model(sample_dataset, temp_output_dir):
    """Test that models are saved correctly."""
    # Create a simple model
    n_cov = sample_dataset.covariates.shape[1]
    n_prots = sample_dataset.X.shape[1]
    model = ProtriderAutoencoder(
        in_dim=n_prots,
        latent_dim=5,
        n_layers=1,
        n_cov=n_cov,
        prot_means=sample_dataset.prot_means_torch,
        presence_absence=False
    )
    model.double()
    
    # Save the model
    checkpoint_path = Path(temp_output_dir) / 'test_model.pt'
    save_model(model, str(checkpoint_path), q=5)
    
    # Check that file exists
    assert checkpoint_path.exists(), "Checkpoint file should be created"
    
    # Check that checkpoint has required keys
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert 'model_state_dict' in checkpoint
    assert 'q' in checkpoint
    assert 'n_layers' in checkpoint
    assert checkpoint['q'] == 5
    assert checkpoint['n_layers'] == 1


def test_load_model(sample_dataset, temp_output_dir):
    """Test that models are loaded correctly."""
    # Create and save a model first
    n_cov = sample_dataset.covariates.shape[1]
    n_prots = sample_dataset.X.shape[1]
    model_original = ProtriderAutoencoder(
        in_dim=n_prots,
        latent_dim=5,
        n_layers=1,
        n_cov=n_cov,
        prot_means=sample_dataset.prot_means_torch,
        presence_absence=False
    )
    model_original.double()
    checkpoint_path = Path(temp_output_dir) / 'test_model.pt'
    save_model(model_original, str(checkpoint_path), q=5)
    
    # Load the model
    config = ProtriderConfig(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        out_dir=temp_output_dir,
        device='cpu'
    )
    
    model_loaded, q_loaded = load_model(sample_dataset, str(checkpoint_path), config)
    
    # Verify model was loaded
    assert model_loaded is not None, "Model should be loaded"
    assert q_loaded == 5, "Latent dimension should match"
    
    # Verify model state matches
    for key in model_original.state_dict():
        assert torch.allclose(
            model_original.state_dict()[key],
            model_loaded.state_dict()[key]
        ), f"State dict key {key} should match"


def test_load_nonexistent_model(sample_dataset, temp_output_dir):
    """Test that loading returns None when model doesn't exist."""
    config = ProtriderConfig(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        out_dir=temp_output_dir,
        device='cpu'
    )
    
    checkpoint_path = Path(temp_output_dir) / 'nonexistent.pt'
    model, q = load_model(sample_dataset, str(checkpoint_path), config)
    
    assert model is None, "Should return None when model doesn't exist"
    assert q is None, "Should return None for q when model doesn't exist"


def test_custom_checkpoint_path(temp_output_dir):
    """Test custom checkpoint path functionality."""
    checkpoint_path = Path(temp_output_dir) / 'custom_checkpoint.pt'
    
    # First run: train and save to custom path
    config1 = ProtriderConfig(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        sample_annotation='sample_data/sample_annotations.tsv',
        out_dir=temp_output_dir,
        checkpoint_path=str(checkpoint_path),
        n_epochs=2,  # Minimal for testing
        device='cpu',
        find_q_method='5'
    )
    
    result1, model_info1 = run(config1)
    
    # Verify checkpoint was saved
    assert checkpoint_path.exists(), "Checkpoint should be saved at custom path"
    
    # Second run: load from custom path
    config2 = ProtriderConfig(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        sample_annotation='sample_data/sample_annotations.tsv',
        out_dir=temp_output_dir,
        checkpoint_path=str(checkpoint_path),
        n_epochs=2,
        device='cpu',
        find_q_method='5'
    )
    
    result2, model_info2 = run(config2)
    
    # Verify model was loaded
    assert model_info1.q == model_info2.q, "Latent dimension should match"


def test_default_checkpoint_behavior(temp_output_dir):
    """Test default checkpoint behavior (auto-save to out_dir)."""
    # First run: train and auto-save
    config1 = ProtriderConfig(
        input_intensities='sample_data/protrider_sample_dataset.tsv',
        sample_annotation='sample_data/sample_annotations.tsv',
        out_dir=temp_output_dir,
        # checkpoint_path not specified - uses default
        n_epochs=2,
        device='cpu',
        find_q_method='5'
    )
    
    result1, model_info1 = run(config1)
    
    # Verify default checkpoint was saved
    default_checkpoint = Path(temp_output_dir) / 'model.pt'
    assert default_checkpoint.exists(), "Default checkpoint should be saved"
    
    # Second run: auto-load from default location
    result2, model_info2 = run(config1)
    assert model_info1.q == model_info2.q, "Should load from default location"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
