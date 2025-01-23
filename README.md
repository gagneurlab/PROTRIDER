# OUTRIDER-prot

FIXME: description

For more information see:
 - FIXME: manuscript link 

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

## Usage

```
protrider --config {config_path} --input_intensities {intensities_csv} --sample_annotation {sample_anno}
```

