"""
Test suite for covariate parsing functionality.

This module contains comprehensive tests for the covariate parsing functions
in protrider.datasets.covariates, covering various scenarios including:
- Numerical/continuous covariates
- Categorical covariates  
- Mixed covariate types
- Missing values and NA indicators
- Edge cases and error conditions
"""

import pytest
import numpy as np
import tempfile
import os
from protrider.datasets.covariates import parse_covariates


class TestParseCovariates:
    """Test class for parse_covariates function."""
    
    def test_parse_continuous_covariates(self, continuous_covariates, covariates_path):
        """Test parsing of continuous/numerical covariates."""
        covariates, centered_covariates = parse_covariates(covariates_path, continuous_covariates)
        
        # Basic shape and type checks
        assert isinstance(covariates, np.ndarray)
        assert isinstance(centered_covariates, np.ndarray)
        assert covariates.shape[0] == 64  # Should match number of samples
        assert centered_covariates.shape[0] == 64
        assert covariates.shape[1] == 1  # AGE is single numerical column
        assert centered_covariates.shape[1] == 1
        
        # Check that covariates have no NaN values
        assert not np.isnan(covariates).any()
        assert not np.isnan(centered_covariates).any()
        
        # Check that centered covariates are actually centered (mean ~0)
        assert abs(np.mean(centered_covariates)) < 1e-10

    def test_parse_categorical_covariates(self, categorical_covariates, covariates_path):
        """Test parsing of categorical covariates."""
        covariates, centered_covariates = parse_covariates(covariates_path, categorical_covariates)
        
        # Basic shape checks
        assert covariates.shape[0] == 64
        assert centered_covariates.shape[0] == 64
        
        # Should have: SEX (2 dummies) + BATCH_RUN (8 dummies) = 10 columns
        expected_cols = 1 + 7  # SEX dummies + BATCH_RUN dummies
        assert covariates.shape[1] == expected_cols
        assert centered_covariates.shape[1] == expected_cols
        
        # No NaN values
        assert not np.isnan(covariates).any()
        assert not np.isnan(centered_covariates).any()

    def test_parse_mixed_covariates(self, covariates_path):
        """Test parsing of mixed numerical and categorical covariates."""
        mixed_covs = ['AGE', 'SEX', 'BATCH_RUN']
        covariates, centered_covariates = parse_covariates(covariates_path, mixed_covs)
        
        # Shape checks
        assert covariates.shape[0] == 64
        assert centered_covariates.shape[0] == 64
        
        # Should have: 1 numerical (AGE) + 2 categorical (SEX: 2 dummies) + (BATCH_RUN: 8 dummies) = 11 columns
        expected_cols = 1 + 1 + 7  # AGE + SEX dummies + BATCH_RUN dummies
        assert covariates.shape[1] == expected_cols
        assert centered_covariates.shape[1] == expected_cols
        
        # No NaN values
        assert not np.isnan(covariates).any()
        assert not np.isnan(centered_covariates).any()

    def test_parse_covariates_with_na_values(self):
        """Test parsing of covariates with missing values."""
        # Create temporary file with missing values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("sample_ID\tAGE\tSEX\tBATCH\n")
            f.write("sample_1\t25\tM\tA\n")
            f.write("sample_2\t\tF\tB\n")  # Missing AGE
            f.write("sample_3\t30\t\tA\n")  # Missing SEX
            f.write("sample_4\t35\tM\t\n")  # Missing BATCH
            temp_file = f.name
        
        try:
            covariates, centered_covariates = parse_covariates(temp_file, ['AGE', 'SEX', 'BATCH'])
            
            # Should have: 1 numerical + 1 NA indicator + 2 categorical + 1 NA indicator + 2 categorical + 1 NA indicator
            # AGE (1) + AGE_NA (1) + SEX dummies (1) + SEX_NA (1) + BATCH dummies (1) + BATCH_NA (1) = 6
            expected_cols = 6
            assert covariates.shape[1] == expected_cols
            assert centered_covariates.shape[1] == expected_cols
            
            # No NaN values in output
            assert not np.isnan(covariates).any()
            assert not np.isnan(centered_covariates).any()
            
        finally:
            os.unlink(temp_file)

    def test_parse_covariates_insufficient_variation(self):
        """Test parsing of covariates with insufficient variation (should be skipped)."""
        # Create temporary file with constant covariate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("sample_ID\tCONSTANT\tVARIABLE\n")
            f.write("sample_1\tA\tX\n")
            f.write("sample_2\tA\tY\n")
            f.write("sample_3\tA\tX\n")
            temp_file = f.name
        
        try:
            covariates, centered_covariates = parse_covariates(temp_file, ['CONSTANT', 'VARIABLE'])
            
            # Should only have VARIABLE covariate (1 dummies), CONSTANT should be skipped
            assert covariates.shape[1] == 1  # Only VARIABLE dummies
            assert centered_covariates.shape[1] == 1
            
        finally:
            os.unlink(temp_file)

    def test_parse_covariates_no_valid_covariates(self):
        """Test case where no valid covariates remain after filtering."""
        # Create temporary file with only constant/invalid covariates
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("sample_ID\tCONSTANT1\tCONSTANT2\n")
            f.write("sample_1\tA\tX\n")
            f.write("sample_2\tA\tX\n")
            f.write("sample_3\tA\tX\n")
            temp_file = f.name
        
        try:
            # Should raise an error when no valid covariates are found
            with pytest.raises(ValueError, match="No valid covariates found"):
                parse_covariates(temp_file, ['CONSTANT1', 'CONSTANT2'])
            
        finally:
            os.unlink(temp_file)

    def test_parse_covariates_csv_format(self):
        """Test parsing of CSV format annotation file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sample_ID,AGE,SEX\n")
            f.write("sample_1,25,M\n")
            f.write("sample_2,30,F\n")
            f.write("sample_3,35,M\n")
            temp_file = f.name
        
        try:
            covariates, centered_covariates = parse_covariates(temp_file, ['AGE', 'SEX'])
            
            # Should work with CSV format
            assert covariates.shape[0] == 3
            assert covariates.shape[1] == 2  # AGE (1) + SEX dummies (1)
            assert not np.isnan(covariates).any()
            
        finally:
            os.unlink(temp_file)


class TestCovariateProperties:
    """Test class for verifying properties of processed covariates."""
    
    def test_categorical_encoding_properties(self, categorical_covariates, covariates_path):
        """Test properties of categorical encoding (drop-first encoding)."""
        covariates, centered_covariates = parse_covariates(covariates_path, categorical_covariates)
        
        # For categorical variables, covariates and centered_covariates should be identical
        # (no centering applied to categorical variables)
        np.testing.assert_array_equal(covariates, centered_covariates)
        
        # Check that categorical encoding is binary (0 or 1)
        unique_values = np.unique(covariates)
        assert set(unique_values).issubset({0, 1})
        
        # With drop_first=True encoding, row sums vary from 0 to number of categorical variables
        # We have 2 categorical variables (SEX, BATCH_RUN)
        # Each sample belongs to one category per variable, but first categories are dropped (represented as all zeros)
        row_sums = np.sum(covariates, axis=1)
        assert np.all(row_sums >= 0)  # Row sums should be non-negative
        assert np.all(row_sums <= 2)  # Row sums should not exceed number of categorical variables

    def test_numerical_centering_properties(self, continuous_covariates, covariates_path):
        """Test properties of numerical covariate centering."""
        covariates, centered_covariates = parse_covariates(covariates_path, continuous_covariates)
        
        # Original covariates should not be centered
        assert abs(np.mean(covariates)) > 1  # AGE mean should be around 30-40
        
        # Centered covariates should have mean near zero
        assert abs(np.mean(centered_covariates)) < 1e-10
        
        # Variance should be the same for both
        np.testing.assert_allclose(np.var(covariates), np.var(centered_covariates), rtol=1e-10)

    def test_na_indicator_properties(self):
        """Test properties of NA indicator variables."""
        # Create temporary file with systematic missing pattern
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("sample_ID\tAGE\tSEX\n")
            f.write("sample_1\t25\tM\n")
            f.write("sample_2\t\tF\n")  # Missing AGE
            f.write("sample_3\t30\t\n")  # Missing SEX
            f.write("sample_4\t\t\n")    # Missing both
            temp_file = f.name
        
        try:
            covariates, centered_covariates = parse_covariates(temp_file, ['AGE', 'SEX'])
            
            # Expected structure: AGE(1) + SEX(1) + AGE_NA(1) + SEX_NA(1) = 4 columns
            assert covariates.shape[1] == 4
            
            # Column structure: [AGE, SEX_F, SEX_M, AGE_NA, SEX_NA]
            # Order is: numerical, categorical, na_indicators
            age_na_col = 2  # AGE_NA comes after categorical variables
            sex_na_col = 3  # SEX_NA is last
            
            # NA indicators should be binary (0 or 1)
            assert set(np.unique(covariates[:, age_na_col])).issubset({0, 1})
            assert set(np.unique(covariates[:, sex_na_col])).issubset({0, 1})
            
            # Check that NA indicators correctly identify missing values
            # Sample 2 (index 1) should have AGE_NA = 1
            assert covariates[1, age_na_col] == 1
            # Sample 3 (index 2) should have SEX_NA = 1  
            assert covariates[2, sex_na_col] == 1
            # Sample 4 (index 3) should have both NA indicators = 1
            assert covariates[3, age_na_col] == 1
            assert covariates[3, sex_na_col] == 1
            
            # Check that non-missing values have NA indicators = 0
            assert covariates[0, age_na_col] == 0  # Sample 1 has AGE
            assert covariates[0, sex_na_col] == 0  # Sample 1 has SEX
            
        finally:
            os.unlink(temp_file)


class TestErrorHandling:
    """Test class for error conditions and edge cases."""
    
    def test_parse_covariates_error_cases(self):
        """Test error cases for covariate parsing."""
        
        # Test with None sa_file
        with pytest.raises(ValueError, match="Sample annotation file is required"):
            parse_covariates(None, ['AGE'])
        
        # Test with None cov_used
        with pytest.raises(ValueError, match="Covariates to use must be specified"):
            parse_covariates('sample_data/sample_annotations.tsv', None)
        
        # Test with unsupported file type
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                parse_covariates(temp_file, ['AGE'])
        finally:
            os.unlink(temp_file)

    def test_parse_covariates_nonexistent_columns(self, covariates_path):
        """Test behavior with non-existent covariate columns."""
        with pytest.raises(KeyError):
            parse_covariates(covariates_path, ['NONEXISTENT_COLUMN'])

    def test_parse_covariates_empty_file(self):
        """Test behavior with empty annotation file."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sample_ID,AGE\n")  # Header only, no data
            temp_file = f.name
        
        try:
            # Should raise an error when no valid covariates are found (due to no data)
            with pytest.raises(ValueError, match="No valid covariates found"):
                parse_covariates(temp_file, ['AGE'])
            
        finally:
            os.unlink(temp_file)


class TestDataConsistency:
    """Test class for data consistency and integration with other components."""
    
    def test_consistency_with_protein_data(self, covariates_path, protein_intensities_path, protein_intensities_index_col):
        """Test that covariate parsing is consistent with protein intensity data."""
        from protrider.datasets.protein_intensities import read_protein_intensities
        
        # Read protein intensities to get sample count
        protein_data = read_protein_intensities(protein_intensities_path, protein_intensities_index_col)
        
        # Parse covariates
        covariates, centered_covariates = parse_covariates(covariates_path, ['AGE', 'SEX'])
        
        # Sample counts should match
        assert covariates.shape[0] == protein_data.shape[0]
        assert centered_covariates.shape[0] == protein_data.shape[0]

    def test_output_dtype_consistency(self, continuous_covariates, covariates_path):
        """Test that output data types are consistent for numerical covariates only."""
        # Test with only numerical covariates to ensure dtype consistency
        covariates, centered_covariates = parse_covariates(covariates_path, continuous_covariates)
        
        # Both outputs should be numpy arrays
        assert isinstance(covariates, np.ndarray)
        assert isinstance(centered_covariates, np.ndarray)
        
        # Original covariates preserve original dtype (int64 for AGE)
        # Centered covariates become float64 due to arithmetic operations
        assert np.issubdtype(covariates.dtype, np.integer) or np.issubdtype(covariates.dtype, np.floating)
        assert np.issubdtype(centered_covariates.dtype, np.floating)

    def test_mixed_dtype_behavior(self, covariates_path):
        """Test data type behavior with mixed numerical and categorical covariates."""
        covariates, centered_covariates = parse_covariates(covariates_path, ['AGE', 'SEX'])
        
        # With mixed types, dtype will be determined by numpy's type promotion rules
        assert isinstance(covariates, np.ndarray)
        assert isinstance(centered_covariates, np.ndarray)
        
        # Both should be numeric (either int or float)
        assert np.issubdtype(covariates.dtype, np.number)
        assert np.issubdtype(centered_covariates.dtype, np.number)
