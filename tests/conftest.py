import pytest
import logging
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logger():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def covariates_path():
    return Path('sample_data/sample_annotations.tsv')


@pytest.fixture
def protein_intensities_path():
    return Path('sample_data/protrider_sample_dataset.tsv')


@pytest.fixture
def protein_intensities_index_col():
    return 'protein_ID'


@pytest.fixture
def categorical_covariates():
    return ['BATCH_RUN', 'SEX']


@pytest.fixture
def continuous_covariates():
    return ['AGE']
