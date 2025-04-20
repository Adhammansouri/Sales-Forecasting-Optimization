import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.utils.config import Config, get_config
from src.utils.logger import get_logger
from src.utils.validation import DataValidationConfig, DataValidator
from src.utils.progress import PipelineProgress

from src.data.generate_synthetic_data import generate_synthetic_sales_data, save_data
from src.data.preprocess_data import preprocess_data
from src.models.train_model import SalesForecaster
from src.models.predict import SalesPredictor
from src.visualization.visualize import SalesVisualizer

# Set up logging
logger = get_logger(__name__, log_dir='logs')

def setup_data_validator(config: Config) -> DataValidator:
    """Set up data validator with configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured DataValidator instance
    """
    validation_config = DataValidationConfig(
        required_columns=['date', 'sales', 'store_id', 'product_id'],
        date_column='date',
        numeric_columns=['sales', 'price', 'promotion'],
        categorical_columns=['store_id', 'product_id', 'holiday'],
        min_date='2020-01-01',  # Adjust based on your needs
        max_date=datetime.now().strftime('%Y-%m-%d')
    )
    return DataValidator(validation_config)

def main():
    """Main function to run the entire sales forecasting pipeline."""
    try:
        # Load configuration
        config = get_config()
        if not config:
            logger.error("Could not load configuration")
            return 1
            
        # Initialize progress tracker
        progress = PipelineProgress(output_dir='progress')
        progress.start_stage('initialization')
        
        # Create necessary directories
        for dir_path in [
            config.data.raw_data_path,
            config.data.processed_data_path,
            config.models.model_dir,
            config.visualization.output_dir
        ]:
            Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
        
        progress.complete_stage('initialization')
        
        # Generate synthetic data if needed
        progress.start_stage('data_generation')
        if not os.path.exists(config.data.raw_data_path):
            logger.info("Generating synthetic data...")
            df = generate_synthetic_sales_data()
            save_data(df, config.data.raw_data_path)
            logger.info(f"Synthetic data saved to {config.data.raw_data_path}")
        else:
            logger.info("Using existing raw data")
        progress.complete_stage('data_generation')
        
        # Load and validate data
        progress.start_stage('data_validation')
        raw_data = pd.read_csv(config.data.raw_data_path)
        validator = setup_data_validator(config)
        validation_results = validator.validate_all(raw_data)
        
        if not validator.is_valid():
            logger.error("Data validation failed")
            progress.fail_stage('data_validation', "Data validation failed")
            return 1
            
        progress.complete_stage('data_validation', metrics=validation_results)
        
        # Preprocess data
        progress.start_stage('preprocessing')
        processed_df = preprocess_data(
            input_filepath=config.data.raw_data_path,
            output_filepath=config.data.processed_data_path
        )
        progress.complete_stage('preprocessing')
        
        # Train models
        progress.start_stage('model_training')
        forecaster = SalesForecaster(model_dir=config.models.model_dir)
        
        try:
            # Train XGBoost
            logger.info("Training XGBoost model...")
            xgb_metrics = forecaster.train_xgboost(
                processed_df,
                target_column=config.features.target_column
            )
            
            # Train Prophet
            logger.info("Training Prophet model...")
            prophet_metrics = forecaster.train_prophet(
                processed_df,
                target_column=config.features.target_column
            )
            
            # Train LSTM
            logger.info("Training LSTM model...")
            lstm_metrics = forecaster.train_lstm(
                processed_df,
                target_column=config.features.target_column
            )
            
            # Save all models
            forecaster.save_models()
            
            training_metrics = {
                'xgboost': xgb_metrics,
                'prophet': prophet_metrics,
                'lstm': lstm_metrics
            }
            progress.complete_stage('model_training', metrics=training_metrics)
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            progress.fail_stage('model_training', str(e))
            return 1
        
        # Generate predictions
        progress.start_stage('prediction')
        try:
            predictor = SalesPredictor(model_dir=config.models.model_dir)
            predictions = predictor.ensemble_predict(
                processed_df['date'].max(),
                periods=30,
                weights=config.ensemble.weights
            )
            predictions.to_csv(config.data.predictions_path, index=False)
            logger.info(f"Predictions saved to {config.data.predictions_path}")
            progress.complete_stage('prediction')
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            progress.fail_stage('prediction', str(e))
            return 1
        
        # Generate visualizations
        progress.start_stage('visualization')
        try:
            visualizer = SalesVisualizer(output_dir=config.visualization.output_dir)
            
            visualizer.plot_sales_trend(processed_df)
            visualizer.plot_seasonal_patterns(processed_df)
            visualizer.plot_product_performance(processed_df)
            visualizer.plot_regional_analysis(processed_df)
            visualizer.plot_forecast_comparison(processed_df, predictions)
            
            if training_metrics:
                visualizer.plot_model_performance(training_metrics)
                
            progress.complete_stage('visualization')
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            progress.fail_stage('visualization', str(e))
            return 1
        
        # Complete pipeline
        progress.complete_pipeline()
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if 'progress' in locals():
            progress.complete_pipeline()
        return 1

if __name__ == '__main__':
    exit(main())