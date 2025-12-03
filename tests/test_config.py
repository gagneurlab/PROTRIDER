"""Tests for ProtriderConfig dataclass and config loading."""
import pytest
import tempfile
import yaml
from pathlib import Path
from protrider import ProtriderConfig, load_config


class TestProtriderConfig:
    """Test ProtriderConfig dataclass."""
    
    def test_minimal_config(self):
        """Test creating config with only required fields."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.out_dir == "output"
        assert config.input_intensities == "data.csv"
        # Check defaults are set
        assert config.seed == 42
        assert config.n_epochs == 100
        assert config.lr == 1e-4
        assert config.device == "gpu"
    
    def test_config_with_all_fields(self):
        """Test creating config with all fields specified."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv",
            index_col="gene_id",
            sample_annotation="annotations.csv",
            max_allowed_NAs_per_protein=0.5,
            log_func_name="log2",
            cov_used=["SEX", "AGE"],
            seed=123,
            inj_freq=1e-2,
            inj_mean=2.5,
            inj_sd=1.2,
            gs_epochs=50,
            autoencoder_training=False,
            n_layers=2,
            n_epochs=200,
            lr=1e-3,
            batch_size=32,
            find_q_method="gs",
            init_pca=False,
            h_dim=64,
            presence_absence=True,
            lambda_presence_absence=0.3,
            cross_val=True,
            n_folds=5,
            early_stopping_patience=25,
            early_stopping_min_delta=0.001,
            fit_every_fold=True,
            pval_dist="gaussian",
            pval_adj="bh",
            pval_sided="left",
            pseudocount=0.05,
            outlier_threshold=0.05,
            report_all=False,
            verbose=True,
            device="cpu"
        )
        assert config.log_func_name == "log2"
        assert config.cov_used == ["SEX", "AGE"]
        assert config.n_layers == 2
        assert config.batch_size == 32
        assert config.pval_dist == "gaussian"
    
    def test_attribute_access(self):
        """Test that config fields are accessible as attributes."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        # Should be able to access with dot notation
        assert config.n_layers == 1
        assert config.find_q_method == "OHT"
        assert config.pval_dist == "t"
        assert hasattr(config, "out_dir")
        assert hasattr(config, "lr")
    
    def test_type_checking(self):
        """Test that types are correctly set."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert isinstance(config.lr, float)
        assert isinstance(config.n_epochs, int)
        assert isinstance(config.verbose, bool)
        assert isinstance(config.seed, int) or config.seed is None


class TestConfigValidation:
    """Test ProtriderConfig validation."""
    
    def test_negative_learning_rate(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="lr must be positive"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                lr=-0.1
            )
    
    def test_zero_learning_rate(self):
        """Test that zero learning rate raises error."""
        with pytest.raises(ValueError, match="lr must be positive"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                lr=0
            )
    
    def test_invalid_max_na_too_high(self):
        """Test that max_allowed_NAs_per_protein > 1 raises error."""
        with pytest.raises(ValueError, match="max_allowed_NAs_per_protein must be between 0 and 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                max_allowed_NAs_per_protein=1.5
            )
    
    def test_invalid_max_na_negative(self):
        """Test that negative max_allowed_NAs_per_protein raises error."""
        with pytest.raises(ValueError, match="max_allowed_NAs_per_protein must be between 0 and 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                max_allowed_NAs_per_protein=-0.1
            )
    
    def test_invalid_outlier_threshold_too_high(self):
        """Test that outlier_threshold > 1 raises error."""
        with pytest.raises(ValueError, match="outlier_threshold must be between 0 and 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                outlier_threshold=1.5
            )
    
    def test_invalid_outlier_threshold_negative(self):
        """Test that negative outlier_threshold raises error."""
        with pytest.raises(ValueError, match="outlier_threshold must be between 0 and 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                outlier_threshold=-0.1
            )
    
    def test_invalid_n_layers(self):
        """Test that n_layers < 1 raises error."""
        with pytest.raises(ValueError, match="n_layers must be at least 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                n_layers=0
            )
    
    def test_invalid_n_epochs(self):
        """Test that n_epochs < 1 raises error."""
        with pytest.raises(ValueError, match="n_epochs must be at least 1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                n_epochs=0
            )
    
    def test_invalid_find_q_method(self):
        """Test that invalid find_q_method raises error."""
        with pytest.raises(ValueError, match="find_q_method must be"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                find_q_method="invalid_method"
            )
    
    def test_valid_find_q_method_oht(self):
        """Test that 'OHT' is a valid find_q_method."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv",
            find_q_method="OHT"
        )
        assert config.find_q_method == "OHT"
    
    def test_valid_find_q_method_gs(self):
        """Test that 'gs' is a valid find_q_method."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv",
            find_q_method="gs"
        )
        assert config.find_q_method == "gs"
    
    def test_valid_find_q_method_integer(self):
        """Test that integer string is a valid find_q_method."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv",
            find_q_method="10"
        )
        assert config.find_q_method == "10"
    
    def test_presence_absence_warning(self):
        """Test that warning is issued when presence_absence=True and n_layers!=1."""
        with pytest.warns(UserWarning, match="Presence absence modeling is only validated with n_layers=1"):
            ProtriderConfig(
                out_dir="output",
                input_intensities="data.csv",
                presence_absence=True,
                n_layers=2
            )


class TestLoadConfig:
    """Test load_config function."""
    
    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        config_dict = {
            'out_dir': 'test_output',
            'input_intensities': 'test_data.csv',
            'index_col': 'protein_ID',
            'n_epochs': 150,
            'lr': 1e-3,
            'seed': 99,
            'verbose': True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.out_dir == 'test_output'
            assert config.input_intensities == 'test_data.csv'
            assert config.n_epochs == 150
            assert config.lr == 1e-3
            assert config.seed == 99
            assert config.verbose is True
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_scientific_notation(self):
        """Test loading config with scientific notation that becomes string."""
        config_dict = {
            'out_dir': 'output',
            'input_intensities': 'data.csv',
            'lr': '1e-4',  # This can happen with YAML
            'inj_freq': '1e-3'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert isinstance(config.lr, float)
            assert config.lr == 1e-4
            assert isinstance(config.inj_freq, float)
            assert config.inj_freq == 1e-3
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent_file.yaml")
    
    def test_load_empty_file(self):
        """Test that loading empty file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Empty configuration file"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_sample_config(self):
        """Test loading the actual sample config.yaml if it exists."""
        config_path = Path('config.yaml')
        if config_path.exists():
            config = load_config(config_path)
            assert config.out_dir == 'output'
            assert config.input_intensities == 'sample_data/protrider_sample_dataset.tsv'
            assert config.index_col == 'protein_ID'
            assert config.n_epochs == 100
            assert config.find_q_method == 'OHT'
    
    def test_load_with_optional_fields(self):
        """Test loading config with optional fields set to None."""
        config_dict = {
            'out_dir': 'output',
            'input_intensities': 'data.csv',
            'sample_annotation': None,
            'batch_size': None,
            'cov_used': None
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.sample_annotation is None
            assert config.batch_size is None
            assert config.cov_used is None
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_list_covariate(self):
        """Test loading config with list of covariates."""
        config_dict = {
            'out_dir': 'output',
            'input_intensities': 'data.csv',
            'cov_used': ['SEX', 'AGE', 'BATCH_RUN']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.cov_used == ['SEX', 'AGE', 'BATCH_RUN']
            assert isinstance(config.cov_used, list)
        finally:
            Path(temp_path).unlink()


class TestConfigDefaults:
    """Test that defaults match expected values from config.yaml."""
    
    def test_preprocessing_defaults(self):
        """Test preprocessing parameter defaults."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.max_allowed_NAs_per_protein == 0.3
        assert config.log_func_name == "log"
    
    def test_model_defaults(self):
        """Test model parameter defaults."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.autoencoder_training is True
        assert config.n_layers == 1
        assert config.n_epochs == 100
        assert config.lr == 1e-4
        assert config.batch_size is None
        assert config.find_q_method == "OHT"
        assert config.init_pca is True
        assert config.h_dim is None
    
    def test_statistical_defaults(self):
        """Test statistical parameter defaults."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.pval_dist == "t"
        assert config.pval_adj == "by"
        assert config.pval_sided == "two-sided"
        assert config.pseudocount == 0.01
    
    def test_reporting_defaults(self):
        """Test reporting parameter defaults."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.outlier_threshold == 0.1
        assert config.report_all is True
    
    def test_runtime_defaults(self):
        """Test runtime parameter defaults."""
        config = ProtriderConfig(
            out_dir="output",
            input_intensities="data.csv"
        )
        assert config.verbose is False
        assert config.device == "gpu"
        assert config.seed == 42
