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
- `cov_used`: List of column names contained in the sample annotation file to be included as known covariates.
- `find_q_method`: Method to determine latent space dimension of autoencoder.
- `pval_dist`: Distribution (Gaussian or Student's t-test) for P-value calculation.

### Running PROTRIDER from the command line

Run PROTRIDER using the following command: 

```
protrider --config <config_path> --input_intensities <intensities_path> --sample_annotation <sample_anno_path>
```
