# PROTRIDER Test Suite

The test suite is organized into multiple modules for better maintainability:

## Test Modules

### `test_pipeline_standard.py` (5 tests)
Tests for standard (non-CV) pipeline execution:
- File path inputs
- DataFrame inputs  
- No covariates
- Wide format output
- Long format output

### `test_pipeline_cv.py` (4 tests)
Tests for cross-validation modes:
- K-fold cross-validation
- Leave-one-out cross-validation (LOOCV)
- Early stopping in CV
- Fit every fold option

### `test_pipeline_config.py` (4 tests)
Tests for configuration validation:
- Missing input validation
- Invalid latent dimension method
- Negative epochs
- Invalid NA threshold

### `test_pipeline_features.py` (14 tests)
Tests for advanced configuration options:
- Log transformations (log, log2, log10, none)
- P-value distributions (gaussian, t)
- P-value adjustment methods (bh, by)
- Latent dimension methods (fixed, OHT, grid search)
- NA thresholds
- Batch size and learning rates
- PCA initializationww
- Outlier thresholds
- Presence/absence modeling

### `test_pipeline_misc.py` (10 tests)
Tests for output consistency and edge cases:
- P-value range validation
- Fold change consistency
- Name preservation (samples, proteins)
- Single/multiple covariates
- Custom pseudocount values
- Seed handling and reproducibility
- Configuration save/load
- Report options

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_pipeline_standard.py

# Run specific test
pytest tests/test_pipeline_cv.py::TestPipelineCrossValidation::test_run_with_kfold_cv

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=protrider
```

## Test Strategy

Most tests use fixed latent dimensions (e.g., `find_q_method='5'`) to:
1. Speed up test execution (no hyperparameter search)
2. Ensure reproducibility and deterministic results
3. Focus on testing pipeline logic, not dimension selection algorithms

Tests that verify automatic latent dimension selection (OHT, gs) are included to ensure these methods work correctly.

## Fixtures

Common fixtures are defined in `conftest.py`:
- `protein_intensities_path` - Sample protein data
- `covariates_path` - Sample annotations
- `protein_intensities_index_col` - Index column name
