import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SalesForecaster:
    """Class for training and saving sales forecasting models."""
    
    def __init__(self, model_dir='models'):
        """Initialize the forecaster.
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = model_dir
        self.models = {
            'xgboost': None,
            'prophet': None,
            'lstm': None
        }
        self.scalers = {}
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train_xgboost(self, df, target_column='units_sold', test_size=0.2):
        """Train an XGBoost regression model.
        
        Args:
            df: DataFrame with processed features
            target_column: Target column name
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"Training XGBoost model with target: {target_column}")
        
        # Prepare data
        df = df.copy()
        df = df.sort_values('date')
        
        # Split into train and test sets
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Define features (exclude date and target)
        features = [col for col in df.columns if col not in ['date', target_column]]
        
        # Train model
        model = XGBRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(
            train_df[features],
            train_df[target_column],
            eval_set=[(test_df[features], test_df[target_column])],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Make predictions on test set
        y_pred = model.predict(test_df[features])
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(test_df[target_column], y_pred),
            'rmse': np.sqrt(mean_squared_error(test_df[target_column], y_pred)),
            'r2': r2_score(test_df[target_column], y_pred)
        }
        
        # Store model and metrics
        self.models['xgboost'] = model
        self.metrics['xgboost'] = metrics
        
        print(f"XGBoost model trained. Metrics: {metrics}")
        return metrics
    
    def train_prophet(self, df, target_column='units_sold', test_size=0.2):
        """Train a Prophet model for time series forecasting.
        
        Args:
            df: DataFrame with processed features
            target_column: Target column name
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"Training Prophet model with target: {target_column}")
        
        # Prepare data for Prophet (needs 'ds' for dates and 'y' for target)
        prophet_df = df[['date', target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Split into train and test sets
        prophet_df = prophet_df.sort_values('ds')
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        # Train model
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(train_df)
        
        # Make predictions on test set
        future = pd.DataFrame({'ds': test_df['ds']})
        forecast = model.predict(future)
        
        # Calculate metrics
        y_pred = forecast['yhat'].values
        metrics = {
            'mae': mean_absolute_error(test_df['y'], y_pred),
            'rmse': np.sqrt(mean_squared_error(test_df['y'], y_pred)),
            'r2': r2_score(test_df['y'], y_pred)
        }
        
        # Store model and metrics
        self.models['prophet'] = model
        self.metrics['prophet'] = metrics
        
        print(f"Prophet model trained. Metrics: {metrics}")
        return metrics
    
    def train_lstm(self, df, target_column='units_sold', sequence_length=7, test_size=0.2):
        """Train an LSTM model for time series forecasting.
        
        Args:
            df: DataFrame with processed features
            target_column: Target column name
            sequence_length: Number of time steps to use for each sample
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"Training LSTM model with target: {target_column}")
        
        # Prepare data
        df = df.copy()
        df = df.sort_values('date')
        
        # Define features (exclude date)
        features = [col for col in df.columns if col != 'date']
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, df.columns.get_loc(target_column) - 1])  # -1 because we excluded 'date'
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions on test set
        y_pred = model.predict(X_test).flatten()
        
        # Inverse scale target for metrics calculation
        # Create a dummy array of zeros with same shape as features
        dummy = np.zeros((len(y_test), len(features)))
        # Put the predicted values in the target column position
        dummy[:, df.columns.get_loc(target_column) - 1] = y_pred
        # Inverse transform
        y_pred_inv = scaler.inverse_transform(dummy)[:, df.columns.get_loc(target_column) - 1]
        
        # Do the same for actual values
        dummy = np.zeros((len(y_test), len(features)))
        dummy[:, df.columns.get_loc(target_column) - 1] = y_test
        y_test_inv = scaler.inverse_transform(dummy)[:, df.columns.get_loc(target_column) - 1]
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test_inv, y_pred_inv),
            'rmse': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
            'r2': r2_score(y_test_inv, y_pred_inv)
        }
        
        # Store model, scaler, and metrics
        self.models['lstm'] = model
        self.scalers['lstm'] = scaler
        self.metrics['lstm'] = metrics
        
        print(f"LSTM model trained. Metrics: {metrics}")
        return metrics
    
    def save_models(self):
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            if model is not None:
                if model_name == 'xgboost':
                    model.save_model(os.path.join(self.model_dir, f"{model_name}_model.json"))
                elif model_name == 'prophet':
                    with open(os.path.join(self.model_dir, f"{model_name}_model.pkl"), 'wb') as f:
                        joblib.dump(model, f)
                elif model_name == 'lstm':
                    model.save(os.path.join(self.model_dir, f"{model_name}_model"))
                    with open(os.path.join(self.model_dir, f"{model_name}_scaler.pkl"), 'wb') as f:
                        joblib.dump(self.scalers[model_name], f)
        
        # Save metrics
        with open(os.path.join(self.model_dir, "model_metrics.pkl"), 'wb') as f:
            joblib.dump(self.metrics, f)
            
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load all models from disk."""
        model_paths = {
            'xgboost': os.path.join(self.model_dir, "xgboost_model.json"),
            'prophet': os.path.join(self.model_dir, "prophet_model.pkl"),
            'lstm': os.path.join(self.model_dir, "lstm_model")
        }
        
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                if model_name == 'xgboost':
                    model = XGBRegressor()
                    model.load_model(path)
                    self.models[model_name] = model
                elif model_name == 'prophet':
                    with open(path, 'rb') as f:
                        self.models[model_name] = joblib.load(f)
                elif model_name == 'lstm':
                    self.models[model_name] = tf.keras.models.load_model(path)
                    scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[model_name] = joblib.load(f)
        
        # Load metrics
        metrics_path = os.path.join(self.model_dir, "model_metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = joblib.load(f)
                
        print("Models loaded")

if __name__ == '__main__':
    # Example usage
    forecaster = SalesForecaster()
    # Load data
    df = pd.read_csv('data/processed/processed_sales_data.csv')
    # Train models
    forecaster.train_xgboost(df)
    forecaster.train_prophet(df)
    forecaster.train_lstm(df)
    # Save models
    forecaster.save_models() 