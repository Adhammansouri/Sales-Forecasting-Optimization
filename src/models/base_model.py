from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseForecaster(ABC):
    """Base class for all forecasting models."""
    
    def __init__(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        model_dir: Optional[str] = None
    ):
        """Initialize the base forecaster.
        
        Args:
            model_name: Name of the model
            model_params: Model hyperparameters
            model_dir: Directory to save model artifacts
        """
        self.model_name = model_name
        self.model_params = model_params
        self.model_dir = Path(model_dir) if model_dir else Path('models')
        self.model = None
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of features and target arrays
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        self.metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        return self.metrics
    
    def save(self) -> None:
        """Save model and metrics to disk."""
        model_path = self.model_dir / f"{self.model_name}_model.joblib"
        metrics_path = self.model_dir / f"{self.model_name}_metrics.json"
        
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    
    def load(self) -> None:
        """Load model from disk."""
        model_path = self.model_dir / f"{self.model_name}_model.joblib"
        metrics_path = self.model_dir / f"{self.model_name}_metrics.json"
        
        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Metrics loaded from {metrics_path}")
                
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise 