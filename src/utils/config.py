from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

@dataclass
class ModelConfig:
    xgboost: Dict[str, Any]
    prophet: Dict[str, Any]
    lstm: Dict[str, Any]

@dataclass
class DataConfig:
    raw_data_path: str
    processed_data_path: str
    predictions_path: str
    features: Dict[str, Any]

@dataclass
class VisualizationConfig:
    output_dir: str
    plot_style: str = "seaborn"
    dpi: int = 300

@dataclass
class Config:
    data: DataConfig
    models: ModelConfig
    visualization: VisualizationConfig
    ensemble: Dict[str, Any]
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Config object with all settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            JSONDecodeError: If config file is not valid JSON
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        return cls(
            data=DataConfig(**config_dict['data']),
            models=ModelConfig(**config_dict['models']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            ensemble=config_dict['ensemble']
        )
    
    def save(self, config_path: str) -> None:
        """Save current configuration to a JSON file.
        
        Args:
            config_path: Path where to save the configuration
        """
        config_dict = {
            'data': {
                'raw_data_path': self.data.raw_data_path,
                'processed_data_path': self.data.processed_data_path,
                'predictions_path': self.data.predictions_path,
                'features': self.data.features
            },
            'models': {
                'xgboost': self.models.xgboost,
                'prophet': self.models.prophet,
                'lstm': self.models.lstm
            },
            'visualization': {
                'output_dir': self.visualization.output_dir,
                'plot_style': self.visualization.plot_style,
                'dpi': self.visualization.dpi
            },
            'ensemble': self.ensemble
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration from default or specified path.
    
    Args:
        config_path: Optional path to config file. If None, uses default path.
        
    Returns:
        Config object with all settings
    """
    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent.parent.parent,
            'config',
            'config.json'
        )
    return Config.from_json(config_path) 