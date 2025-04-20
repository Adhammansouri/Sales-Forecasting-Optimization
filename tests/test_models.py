import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.generate_synthetic_data import generate_synthetic_sales_data
from src.data.preprocess_data import preprocess_data
from src.models.train_model import SalesForecaster
from src.models.predict import SalesPredictor
from src.utils.helpers import calculate_metrics

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_synthetic_sales_data(
        start_date='2023-01-01',
        num_days=100
    )

@pytest.fixture
def processed_data(sample_data):
    """Get processed data for testing."""
    return preprocess_data(sample_data)

def test_data_generation(sample_data):
    """Test synthetic data generation."""
    assert isinstance(sample_data, pd.DataFrame)
    assert not sample_data.empty
    assert all(col in sample_data.columns for col in [
        'date', 'product', 'region', 'units_sold', 'unit_price', 'total_sales'
    ])

def test_data_preprocessing(processed_data):
    """Test data preprocessing."""
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert 'date' in processed_data.columns
    assert processed_data['date'].dtype == 'datetime64[ns]'

def test_xgboost_training(processed_data):
    """Test XGBoost model training."""
    forecaster = SalesForecaster(model_dir='test_models')
    forecaster.train_xgboost(processed_data, 'units_sold')
    
    assert forecaster.models['xgboost'] is not None

def test_prophet_training(processed_data):
    """Test Prophet model training."""
    forecaster = SalesForecaster(model_dir='test_models')
    forecaster.train_prophet(processed_data, 'units_sold')
    
    assert forecaster.models['prophet'] is not None

def test_lstm_training(processed_data):
    """Test LSTM model training."""
    forecaster = SalesForecaster(model_dir='test_models')
    forecaster.train_lstm(processed_data, 'units_sold')
    
    assert forecaster.models['lstm'] is not None

def test_model_prediction(processed_data):
    """Test model predictions."""
    # Train models
    forecaster = SalesForecaster(model_dir='test_models')
    forecaster.train_xgboost(processed_data, 'units_sold')
    forecaster.save_models()
    
    # Make predictions
    predictor = SalesPredictor(model_dir='test_models')
    last_date = processed_data['date'].max()
    predictions = predictor.ensemble_predict(last_date, periods=30)
    
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 30
    assert 'ensemble_prediction' in predictions.columns

def test_metrics_calculation():
    """Test metrics calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

if __name__ == '__main__':
    pytest.main([__file__]) 