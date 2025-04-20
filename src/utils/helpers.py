import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def create_directory_structure():
    """Create the project directory structure if it doesn't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'data/predictions',
        'models',
        'notebooks',
        'reports/figures',
        'src/data',
        'src/models',
        'src/visualization',
        'src/utils',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def load_config(config_path='config/config.json'):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        return {}

def save_config(config, config_path='config/config.json'):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics

def generate_date_features(df, date_column='date'):
    """Generate date-based features from a date column."""
    df = df.copy()
    
    # Extract basic components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    
    # Create flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    
    return df

def add_rolling_features(df, column, windows=[7, 30, 90], group_columns=None):
    """Add rolling statistics features."""
    df = df.copy()
    
    if group_columns is None:
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window, min_periods=1).std()
    else:
        for window in windows:
            for group_col in group_columns:
                df[f'{column}_rolling_mean_{window}_{group_col}'] = (
                    df.groupby(group_col)[column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df[f'{column}_rolling_std_{window}_{group_col}'] = (
                    df.groupby(group_col)[column]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
    
    return df

def add_lag_features(df, column, lags=[1, 7, 30], group_columns=None):
    """Add lagged features."""
    df = df.copy()
    
    if group_columns is None:
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    else:
        for lag in lags:
            for group_col in group_columns:
                df[f'{column}_lag_{lag}_{group_col}'] = df.groupby(group_col)[column].shift(lag)
    
    return df

def encode_categorical_features(df, columns=None, drop_original=True):
    """Encode categorical features using one-hot encoding."""
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    df = pd.get_dummies(df, columns=columns, prefix=columns)
    
    return df

def scale_features(df, columns=None, scaler=None):
    """Scale numerical features."""
    from sklearn.preprocessing import StandardScaler
    
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    
    return df, scaler

def save_model(model, model_path, model_type='sklearn'):
    """Save a model to disk."""
    import joblib
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if model_type == 'sklearn':
        joblib.dump(model, model_path)
    elif model_type == 'keras':
        model.save(model_path)
    elif model_type == 'prophet':
        with open(model_path, 'w') as f:
            f.write(model.to_json())

def load_model(model_path, model_type='sklearn'):
    """Load a model from disk."""
    import joblib
    from prophet import Prophet
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if model_type == 'sklearn':
        return joblib.load(model_path)
    elif model_type == 'keras':
        return tf.keras.models.load_model(model_path)
    elif model_type == 'prophet':
        with open(model_path, 'r') as f:
            return Prophet.from_json(f.read())

if __name__ == '__main__':
    # Create project structure when run directly
    create_directory_structure() 