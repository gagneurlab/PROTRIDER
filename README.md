# PROTRIDER

PROTRIDER is an autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

For more information see:

## Installation

### Prerequisites

PROTRIDER was trained and tested using Python 3.14 on a Linux system. The list of required packages for running PROTRIDER can be found in the file requirements.txt.

Using pip and conda environments
We recommend installing and running PROTRIDER on a dedicated conda environment. To create and activate the conda environment run the following commands:

```
conda create --name protrider_env python=3.14
conda activate protrider_env
```

More information on conda environments can be found in Conda's user guide.


To install PROTRIDER run the following command inside the root directory:

```
pip install .
```

To test the installation run 

```
protrider --help
```

## Usage

### Input files

- **Protein intensities**: CSV or tab-separated file
  - **File format**: Columns represent **samples**, rows represent **proteins** (wide format)
  - **DataFrame format**: When passing a pandas DataFrame directly, samples should be **rows**, proteins should be **columns**
  - Example: `sample_data/protrider_sample_dataset.tsv`
- **Sample annotation** (optional): CSV or tab-separated file containing known covariates
  - Format: Each row represents a sample
  - Example: `sample_data/sample_annotations.tsv`

An example dataset can be found in this repository under `sample_data/`. 

### Configuration file

To run PROTRIDER, a configuration file needs to be provided. This can be adapted from the configuration file provided in this code repo (`config.yaml`). User options include

- `out_dir`: Path to the directory to store output files.
- `input_intensities`: Csv input file containing the protein intensities.
- `cov_used`: List of column names contained in the sample annotation file to be included as known covariates.
- `find_q_method`: Method to determine latent space dimension of autoencoder.
- `pval_dist`: Distribution (Gaussian or Student's t-test) for P-value calculation.

### Running PROTRIDER from the command line

Run PROTRIDER using the following command: 

```bash
protrider run --config <config_path>
```

All input and output paths (including `input_intensities`, `sample_annotation`, and `out_dir`) should be specified in the configuration file.

To generate plots with PROTRIDER, use the following command (specify one or more plot types as needed):
```bash
protrider plot --plot_type <plot_types> --config <config_path> --out_dir <out_dir>
```
#### Plot options

You can specify one or more plot types using the `--plot_type` option:

- `training_loss`: Plot training loss history
- `aberrant_per_sample`: Plot number of aberrant proteins per sample
- `pvals`: Plot the p-value plots
- `encoding_dim`: Plot the encoding dimension search plot
- `expected_vs_observed`: Plot expected vs observed protein intensity for a specific protein (requires `--protein_id`)
- `all`: Equivalent to specifying all of the above except `expected_vs_observed`.

Example:
```
protrider plot --plot_type pvals --config <config_path>
```

To plot expected vs observed for a specific protein:
```
protrider plot --plot_type expected_vs_observed --protein_id <protein_id> --config <config_path>
```


#### Output files

PROTRIDER generates several output files in the specified output directory:

- `pvals.csv`: P-values for each protein in each sample
- `pvals_adj.csv`: Adjusted p-values (FDR correction)
- `zscores.csv`: Z-scores for each protein in each sample
- `residuals.csv`: Model residuals
- `log2fc.csv`: Log2 fold changes
- `fc.csv`: Fold changes
- `output.csv`: Combined long-format output (when using `format='long'`)
- `config.yaml`: Saved configuration for reproducibility
- `model_info.yaml`: Model metadata (latent dimension, learning rate, etc.)


### Using PROTRIDER as a Python package

PROTRIDER can also be used directly as a Python package for more flexibility:

```python
from protrider import ProtriderConfig, run_protrider
import pandas as pd

# Option 1: Using file paths
# File format: columns = samples, rows = proteins
config = ProtriderConfig(
    out_dir='output/',
    input_intensities='data/protein_intensities.csv',  # Columns = samples
    sample_annotation='data/sample_annotations.csv',
    index_col='protein_ID',
    cov_used=['AGE', 'SEX'],
    n_epochs=100
)

# Option 2: Using DataFrames directly
# DataFrame format: rows = samples, columns = proteins (transposed from file format!)
protein_df = pd.read_csv('data/protein_intensities.csv', index_col='protein_ID').T
annotation_df = pd.read_csv('data/sample_annotations.csv')

config = ProtriderConfig(
    out_dir='output/',
    input_intensities=protein_df,        # DataFrame: rows = samples, columns = proteins
    sample_annotation=annotation_df,     # DataFrame: rows = samples
    index_col='protein_ID',
    cov_used=['AGE', 'SEX'],
    n_epochs=100
)

# Run PROTRIDER
result, model_info = run_protrider(config)

# Save results in different formats
result.save(config.out_dir, format='wide')  # Individual CSV files (pvals.csv, zscores.csv, etc.)
result.save(config.out_dir, format='long')  # Single combined CSV (output.csv)

# Save model info and config
model_info.save(config.out_dir)  # Saves additional_info.csv, train_losses.csv
config.save(config.out_dir)      # Saves config.yaml

# Generate plots (out_dir is optional - omit to get plot objects without saving)
model_info.plot_training_loss(config.out_dir)
result.plot_aberrant_per_sample(config.out_dir)
hist_plot, qq_plot = result.plot_pvals(config.out_dir)
result.plot_expected_vs_observed('protein_123', config.out_dir)

# Access results as DataFrames for further analysis
pvals = result.df_pvals           # P-values
pvals_adj = result.df_pvals_adj   # Adjusted p-values
zscores = result.df_Z             # Z-scores
residuals = result.df_res         # Residuals
log2fc = result.log2fc            # Log2 fold changes
fc = result.fc                    # Fold changes
```
