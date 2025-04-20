import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.train_model import SalesForecaster
from src.data.preprocess_data import preprocess_data
import joblib
import os

class SalesPredictor:
    def __init__(self, model_dir='models'):
        """Initialize the predictor with trained models."""
        self.forecaster = SalesForecaster(model_dir)
        self.forecaster.load_models()
    
    def prepare_future_data(self, last_date, periods=30):
        """Prepare future dates for prediction."""
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Create a template DataFrame with all necessary features
        future_df = pd.DataFrame({'date': future_dates})
        
        # Add time-based features
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        future_df['week_of_year'] = future_df['date'].dt.isocalendar().week
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['is_month_end'] = future_df['date'].dt.is_month_end.astype(int)
        
        return future_df
    
    def predict_xgboost(self, future_features):
        """Make predictions using XGBoost model."""
        model = self.forecaster.models['xgboost']
        if model is None:
            raise ValueError("XGBoost model not loaded")
        
        predictions = model.predict(future_features)
        return predictions
    
    def predict_prophet(self, future_dates):
        """Make predictions using Prophet model."""
        model = self.forecaster.models['prophet']
        if model is None:
            raise ValueError("Prophet model not loaded")
        
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        return forecast['yhat'].values
    
    def predict_lstm(self, future_features):
        """Make predictions using LSTM model."""
        model = self.forecaster.models['lstm']
        if model is None:
            raise ValueError("LSTM model not loaded")
        
        # Reshape for LSTM
        X_reshaped = future_features.values.reshape((future_features.shape[0], 1, future_features.shape[1]))
        predictions = model.predict(X_reshaped).flatten()
        return predictions
    
    def ensemble_predict(self, last_date, periods=30, weights=None):
        """Make ensemble predictions using all available models."""
        # Prepare future data
        future_df = self.prepare_future_data(last_date, periods)
        
        predictions = {}
        ensemble_pred = np.zeros(periods)
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'xgboost': 0.4,
                'prophet': 0.3,
                'lstm': 0.3
            }
        
        # Get predictions from each model
        try:
            if self.forecaster.models['xgboost'] is not None:
                predictions['xgboost'] = self.predict_xgboost(future_df)
                ensemble_pred += weights['xgboost'] * predictions['xgboost']
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
        
        try:
            if self.forecaster.models['prophet'] is not None:
                predictions['prophet'] = self.predict_prophet(future_df['date'])
                ensemble_pred += weights['prophet'] * predictions['prophet']
        except Exception as e:
            print(f"Error in Prophet prediction: {e}")
        
        try:
            if self.forecaster.models['lstm'] is not None:
                predictions['lstm'] = self.predict_lstm(future_df)
                ensemble_pred += weights['lstm'] * predictions['lstm']
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': future_df['date'],
            'ensemble_prediction': ensemble_pred
        })
        
        # Add individual model predictions
        for model_name, preds in predictions.items():
            results[f'{model_name}_prediction'] = preds
        
        return results

def main():
    """Main function to generate predictions."""
    # Load the last date from processed data
    processed_data = pd.read_csv('data/processed/processed_sales_data.csv')
    last_date = pd.to_datetime(processed_data['date']).max()
    
    # Initialize predictor
    predictor = SalesPredictor()
    
    # Generate predictions for next 30 days
    predictions = predictor.ensemble_predict(last_date, periods=30)
    
    # Save predictions
    output_path = 'data/predictions/forecast.csv'
    os.makedirs('data/predictions', exist_ok=True)
    predictions.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    print("\nSample predictions:")
    print(predictions.head())

if __name__ == '__main__':
    main() 