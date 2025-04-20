import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from xgboost import XGBRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import json
import os

class SalesForecaster:
    def __init__(self, model_dir='models'):
        """Initialize the forecaster with model storage directory."""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {
            'xgboost': None,
            'prophet': None,
            'lstm': None
        }
        
        self.metrics = {}
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        print("Training XGBoost model...")
        
        params = {
            'n_estimators': 1000,
            'max_depth': 7,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        self.models['xgboost'] = model
        return model
    
    def train_prophet(self, df, target_col='units_sold'):
        """Train Prophet model."""
        print("Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        model.fit(prophet_df)
        self.models['prophet'] = model
        return model
    
    def train_lstm(self, X_train, y_train, sequence_length=30):
        """Train LSTM model."""
        print("Training LSTM model...")
        
        # Reshape data for LSTM [samples, time steps, features]
        X_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(100, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        model.fit(
            X_reshaped, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        self.models['lstm'] = model
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate model performance."""
        model = self.models[model_name]
        
        if model_name == 'xgboost':
            y_pred = model.predict(X_test)
        elif model_name == 'lstm':
            X_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_reshaped).flatten()
        else:
            # Prophet evaluation needs different handling
            return
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def save_models(self):
        """Save trained models and their metrics."""
        # Save XGBoost model
        if self.models['xgboost'] is not None:
            joblib.dump(self.models['xgboost'],
                       os.path.join(self.model_dir, 'xgboost_model.joblib'))
        
        # Save Prophet model
        if self.models['prophet'] is not None:
            with open(os.path.join(self.model_dir, 'prophet_model.json'), 'w') as f:
                f.write(self.models['prophet'].to_json())
        
        # Save LSTM model
        if self.models['lstm'] is not None:
            self.models['lstm'].save(os.path.join(self.model_dir, 'lstm_model'))
        
        # Save metrics
        with open(os.path.join(self.model_dir, 'model_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def load_models(self):
        """Load saved models."""
        try:
            # Load XGBoost model
            xgboost_path = os.path.join(self.model_dir, 'xgboost_model.joblib')
            if os.path.exists(xgboost_path):
                self.models['xgboost'] = joblib.load(xgboost_path)
                print("XGBoost model loaded successfully")
            
            # Load Prophet model
            prophet_path = os.path.join(self.model_dir, 'prophet_model.json')
            if os.path.exists(prophet_path):
                with open(prophet_path, 'r') as f:
                    self.models['prophet'] = Prophet.from_json(f.read())
                print("Prophet model loaded successfully")
            
            # Load LSTM model
            lstm_path = os.path.join(self.model_dir, 'lstm_model')
            if os.path.exists(lstm_path):
                self.models['lstm'] = tf.keras.models.load_model(lstm_path)
                print("LSTM model loaded successfully")
            
            # Load metrics
            metrics_path = os.path.join(self.model_dir, 'model_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                print("Model metrics loaded successfully")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

def main():
    """Main function to train and evaluate models."""
    try:
        from src.data.preprocess_data import load_data, preprocess_data, prepare_data_for_training
        
        # Load and preprocess data
        df = preprocess_data()
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_data_for_training(df)
        
        # Initialize forecaster
        forecaster = SalesForecaster()
        
        # Train models
        forecaster.train_xgboost(X_train, y_train)
        forecaster.train_prophet(df)
        forecaster.train_lstm(X_train, y_train)
        
        # Evaluate models
        for model_name in ['xgboost', 'lstm']:
            metrics = forecaster.evaluate_model(model_name, X_test, y_test)
            print(f"\n{model_name.upper()} Model Metrics:")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"R2 Score: {metrics['r2']:.2f}")
        
        # Save models
        forecaster.save_models()
        
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("Please ensure all required packages are installed using 'pip install -r requirements.txt'")
        raise
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 