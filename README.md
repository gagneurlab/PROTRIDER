# PROTRIDER

PROTRIDER is an autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

For more information see:

## Installation

### Prerequisites

PROTRIDER was trained and tested using Python 3.8 on a Linux system. The list of required packages for running PROTRIDER can be found in the file requirements.txt.

Using pip and conda environments
We recommend installing and running PROTRIDER on a dedicated conda environment. To create and activate the conda environment run the following commands:

```
conda create --name protrider_env python=3.13
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

- Experimental protein intensities as csv or tab file, in which the columns represent samples and the rows represent proteins.
- Optional: sample annotation file containing known covariates to be passed to the model.

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

```
protrider run --config <config_path> --input_intensities <intensities_path> --sample_annotation <sample_anno_path> --out_dir <out_dir>
```
To generate plots with PROTRIDER, use the following command (specify one or more plot types as needed):
```
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
