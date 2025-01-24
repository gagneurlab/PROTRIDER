# OUTRIDER-prot

OUTRIDER-prot is a autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

For more information see:

## Installation

### Prerequisites

OUTRIDER-prot was trained and tested using Python 3.8 on a Linux system. The list of required packages for running OUTRIDER-prot can be found in the file requirements.txt.

Using pip and conda environments
We recommend to install and run Spectralis on a dedicated conda environment. To create and activate the conda environment run the following commands:

```
conda create --name outrider_prot_env python=3.8
conda activate outrider_prot_env
```

More information on conda environments can be found in Conda's user guide.


To install OUTRIDER-prot run the following command inside the root directory:

```
pip install .
```

To test the installation run 

```
outrider_prot --help
```

## Usage

### Input files

- Experimental protein intensities as csv or tab file, in which the columns represent samples and the rows represent proteins.
- Optional: sample annotation file containing known covariates to be passed to the model.

An example dataset can be found in this repository. 

### Configuration file

To run OUTRIDER-prot, a configuration file needs to be provided. This can be adapted from the configuration file provided in this code repo (`config.yaml`). User options include

- `out_dir`: Path to directory to store output files.
- `cov_used`: List of column names contained in the sample annotation file to be included as known covariates.
- `find_q_method`: Method to determine latent space dimension of autoencoder.
- `pval_dist`: Distribution (Gaussian or Student's t-test) for P-value calculation.

### Running OUTRIDER-prot from the command line

Run OUTRIDER-prot using the following command: 

```
protrider --config <config_path> --input_intensities <intensities_path> --sample_annotation <sample_anno_path>
```
