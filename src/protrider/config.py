"""Configuration loading utilities."""
import yaml
import torch
from pathlib import Path
from typing import Union, Callable
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import numpy as np

@dataclass
class ProtriderConfig:
    """Configuration for PROTRIDER pipeline.
    
    Matrix Format Requirements:
    ---------------------------
    input_intensities: File path (str)
        - File format: columns = samples, rows = proteins
    
    sample_annotation: File path (str) or None
        - Format: rows = samples
    """
    
    # I/O paths
    input_intensities: str  # File path only
    index_col: str = "protein_ID"
    out_dir: Optional[str] = None  # File path or None
    sample_annotation: Optional[str] = None  # File path or None
    
    # Preprocessing params
    max_allowed_NAs_per_protein: float = 0.3
    log_func_name: Optional[Literal["log", "log2", "log10"]] = "log"
    
    # Computed fields (set in __post_init__)
    log_func: Optional[Callable] = field(init=False, repr=False, default=None)
    base_fn: Callable = field(init=False, repr=False, default=None)
    device_torch: torch.device = field(init=False, repr=False, default=None)
    
    # Covariates
    cov_used: Optional[List[str]] = None
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Grid search outlier injection params
    inj_freq: float = 1e-3
    inj_mean: float = 3
    inj_sd: float = 1.6
    gs_epochs: int = 100
    
    # Model params
    autoencoder_training: bool = True
    n_layers: int = 1
    n_epochs: int = 100
    lr: float = 1e-4
    batch_size: Optional[int] = None
    find_q_method: str = "OHT"  # "OHT", "gs", or an integer
    init_pca: bool = True
    h_dim: Optional[int] = None
    
    # Presence absence modelling
    presence_absence: bool = False
    lambda_presence_absence: float = 0.5
    
    # Cross-validation parameters
    cross_val: bool = False
    n_folds: Optional[int] = None
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0001
    fit_every_fold: bool = False
    
    # Statistical params
    pval_dist: Literal["gaussian", "t"] = "t"
    pval_adj: Literal["by", "bh"] = "by"
    pval_sided: Literal["two-sided", "left", "right"] = "two-sided"
    pseudocount: float = 0.01
    
    # Reporting params
    outlier_threshold: float = 0.1
    report_all: bool = True
    
    # Runtime params
    verbose: bool = False
    device: Literal["gpu", "cpu"] = "gpu"
    
    def __post_init__(self):
        """Validate configuration after initialization and set computed fields."""
        # Validation
        if self.max_allowed_NAs_per_protein < 0 or self.max_allowed_NAs_per_protein > 1:
            raise ValueError("max_allowed_NAs_per_protein must be between 0 and 1")
        
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        
        if self.outlier_threshold < 0 or self.outlier_threshold > 1:
            raise ValueError("outlier_threshold must be between 0 and 1")
        
        if self.find_q_method not in ["OHT", "gs"] and not self.find_q_method.isdigit():
            raise ValueError("find_q_method must be 'OHT', 'gs', or an integer string")
        
        if self.presence_absence and self.n_layers != 1:
            import warnings
            warnings.warn("Presence absence modeling is only validated with n_layers=1")
        
        # Set log_func and base_fn based on log_func_name
        if self.log_func_name == "log2":
            self.log_func = np.log2
            self.base_fn = lambda x: 2 ** x
        elif self.log_func_name == "log10":
            self.log_func = np.log10
            self.base_fn = lambda x: 10 ** x
        elif self.log_func_name == "log":
            self.log_func = np.log
            self.base_fn = np.exp
        elif self.log_func_name is None:
            self.log_func = None
            self.base_fn = np.exp
        else:
            raise ValueError(f"Log func {self.log_func_name} not supported.")
        
        # Set PyTorch device
        self.device_torch = torch.device("cuda" if (torch.cuda.is_available() and self.device == 'gpu') else "cpu")
    
    def save(self, out_dir: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Only serializable fields (those with init=True) are saved.
        Computed fields like log_func, base_fn, and device_torch are excluded.
        
        Args:
            out_dir: Output directory path where config.yaml will be saved
        """
        import dataclasses
        import logging
        
        logger = logging.getLogger(__name__)
        out_dir = Path(out_dir)
        out_p = out_dir / 'config.yaml'
        
        # Only save fields that are part of __init__ (exclude computed fields)
        config_dict = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.init  # Only include fields that are initialized (excludes computed fields)
        }
        
        with open(out_p, 'w') as f:
            yaml.safe_dump(config_dict, f)
        
        logger.info(f"Saved run config to {out_p}")


def load_config(config_path: Union[str, Path]) -> ProtriderConfig:
    """
    Load PROTRIDER configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ProtriderConfig object with validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    # Handle scientific notation that gets loaded as strings
    if 'lr' in config_dict and isinstance(config_dict['lr'], str):
        config_dict['lr'] = float(config_dict['lr'])
    if 'inj_freq' in config_dict and isinstance(config_dict['inj_freq'], str):
        config_dict['inj_freq'] = float(config_dict['inj_freq'])
    
    # Convert to ProtriderConfig, which will validate the fields
    try:
        config = ProtriderConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    return config
