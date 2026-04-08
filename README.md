# PROTRIDER

PROTRIDER is an autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

Have a look at our [paper](https://doi.org/10.1093/bioinformatics/btaf628) for information about our work.

## Table of Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
  - [Input files](#input-files)
  - [Configuration file](#configuration-file)
  - [Running from the command line](#running-protrider-from-the-command-line)
  - [Output files](#output-files)
  - [Model checkpointing](#model-checkpointing)
  - [Using as a Python package](#using-protrider-as-a-python-package)
- [Citation](#citation)

## Quickstart

```bash
# 1. Install
pip install .

# 2. Run on the included sample data
protrider run --config config.yaml

# 3. Plot results
protrider plot --config config.yaml all
```

Results are written to the directory specified by `out_dir` in `config.yaml` (default: `output/`). The key output file is `protrider_summary.csv`, which contains outlier calls with p-values, z-scores, and fold changes for every sample–protein pair.

## Installation

### Prerequisites

PROTRIDER was trained and tested using Python 3.14 on a Linux system. We recommend installing and running PROTRIDER in a dedicated conda environment:

```bash
conda create --name protrider_env python=3.14
conda activate protrider_env
```

More information on conda environments can be found in [Conda's user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/).

### Install

Run the following command inside the root directory:

```bash
pip install .
```

Verify the installation:

```bash
protrider --help
```

## Usage

### Input files

- **Protein intensities**: CSV, TSV, or Parquet file
  - Columns represent **samples**, rows represent **proteins**
  - Example: `sample_data/protrider_sample_dataset.tsv`
- **Sample annotation** (optional): CSV or tab-separated file containing known covariates
  - Each row represents one sample
  - Example: `sample_data/sample_annotations.tsv`

An example dataset is included under `sample_data/`.

### Configuration file

All parameters are set in a YAML configuration file. A template is provided as `config.yaml`. Key options:

| Parameter | Description |
|-----------|-------------|
| `out_dir` | Output directory |
| `input_intensities` | Path to protein intensities file |
| `sample_annotation` | Path to sample annotations file (optional) |
| `index_col` | Column name containing protein IDs |
| `cov_used` | List of covariate column names from the annotation file (optional) |
| `find_q_method` | Method to determine latent dimension: `OHT` (default), `gs`, `bs` (binary search), or an integer |
| `pval_dist` | Distribution for p-value calculation: `t` (default) or `gaussian` |
| `n_epochs` | Number of training epochs (default: `100`) |
| `checkpoint_path` | Path to save/load model checkpoint (optional) |

### Running PROTRIDER from the command line

**Run the pipeline:**

```bash
protrider run --config config.yaml
```

**Generate plots:**

```bash
# All plots
protrider plot --config config.yaml all

# Individual plot types
protrider plot --config config.yaml pvals
protrider plot --config config.yaml aberrant_per_sample
protrider plot --config config.yaml training_loss
protrider plot --config config.yaml encoding_dim

# Expected vs observed for a specific protein
protrider plot --config config.yaml expected_vs_observed --protein_id <protein_id>
```

### Output files

| File | Description |
|------|-------------|
| `protrider_summary.csv` | Long-format summary with outlier calls for all sample–protein pairs |
| `pvals.csv` | Two-sided p-values (samples × proteins) |
| `pvals_adj.csv` | BH/BY-adjusted p-values |
| `pvals_one_sided.csv` | Left-sided p-values |
| `zscores.csv` | Z-scores |
| `residuals.csv` | Model residuals (observed − predicted) |
| `log2fc.csv` | Log2 fold changes |
| `fc.csv` | Fold changes |
| `output.csv` | Autoencoder reconstructed values |
| `processed_input.csv` | Preprocessed input passed to the autoencoder |
| `additional_info.csv` | Model metadata (latent dimension, learning rate, loss) |
| `train_losses.csv` | Per-epoch training loss |
| `fit_parameters.csv` | Per-protein distribution fit parameters |
| `config.yaml` | Saved configuration for reproducibility |

### Model checkpointing

PROTRIDER automatically saves trained models and reuses them in subsequent runs, skipping retraining if a checkpoint exists. By default the model is saved to `<out_dir>/model.pt`.

To use a custom checkpoint location, set `checkpoint_path` in your config:

```yaml
checkpoint_path: models/my_model.pt
```

To force retraining, delete the checkpoint file or point to a new path.

### Using PROTRIDER as a Python package

```python
import protrider

config = protrider.ProtriderConfig(
    out_dir='output/',
    input_intensities='data/protein_intensities.csv',
    sample_annotation='data/sample_annotations.csv',
    index_col='protein_ID',
    cov_used=['AGE', 'SEX'],
    n_epochs=100,
)

# Run
result, model_info, fit_params, gs_result = protrider.run(config)

# Save results
result.save(config.out_dir, format='wide')   # individual CSV files
result.save(config.out_dir, format='long')   # protrider_summary.csv
model_info.save(config.out_dir)
config.save(config.out_dir)

# Generate plots (omit out_dir to get plot objects without saving)
model_info.plot_training_loss(config.out_dir)
result.plot_aberrant_per_sample(config.out_dir)
hist_plot, qq_plot = result.plot_pvals(config.out_dir)
result.plot_expected_vs_observed('protein_123', config.out_dir)

# Access results as DataFrames
result.df_pvals        # p-values
result.df_pvals_adj    # adjusted p-values
result.df_Z            # z-scores
result.df_res          # residuals
result.log2fc          # log2 fold changes
result.fc              # fold changes
```

## Citation

If you use PROTRIDER, please cite:

```bibtex
@article{10.1093/bioinformatics/btaf628,
    author = {Klaproth-Andrade, Daniela and Scheller, Ines F and Tsitsiridis, Georgios and Loipfinger, Stefan and Mertes, Christian and Smirnov, Dmitrii and Prokisch, Holger and Yépez, Vicente A and Gagneur, Julien},
    title = {PROTRIDER: Protein abundance outlier detection from mass spectrometry-based proteomics data with a conditional autoencoder},
    journal = {Bioinformatics},
    pages = {btaf628},
    year = {2025},
    month = {11},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf628},
    url = {https://doi.org/10.1093/bioinformatics/btaf628},
}
```
