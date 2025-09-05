from protrider.datasets.covariates import parse_covariates
from protrider.datasets.protein_intensities import read_protein_intensities
import pandas as pd


def test_read_protein_intensities(protein_intensities_path, protein_intensities_index_col):
    df = read_protein_intensities(
        protein_intensities_path, protein_intensities_index_col)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'sampleID'
    assert df.shape == (64, 200)


def test_parse_categorical_covariates(categorical_covariates, covariates_path, protein_intensities_path, protein_intensities_index_col):
    """Test basic integration between protein intensities and categorical covariates."""
    protein_intensities = read_protein_intensities(
        protein_intensities_path, protein_intensities_index_col)
    covariates, centered_covariates_noNA = parse_covariates(covariates_path, categorical_covariates)
    assert covariates.shape[0] == protein_intensities.shape[0]
    assert centered_covariates_noNA.shape[0] == protein_intensities.shape[0]
