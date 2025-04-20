import os
import json
from datetime import datetime

from src.data.generate_synthetic_data import generate_synthetic_sales_data, save_data
from src.data.preprocess_data import preprocess_data
from src.models.train_model import SalesForecaster
from src.models.predict import SalesPredictor
from src.visualization.visualize import SalesVisualizer
from src.utils.helpers import create_directory_structure, load_config

def main():
    """Main function to run the entire sales forecasting pipeline."""
    print("Starting Sales Forecasting Pipeline...")
    start_time = datetime.now()
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Load configuration
    print("\n2. Loading configuration...")
    config = load_config()
    if not config:
        print("Error: Could not load configuration. Exiting.")
        return
    
    # Generate synthetic data if it doesn't exist
    if not os.path.exists(config['data']['raw_data_path']):
        print("\n3. Generating synthetic data...")
        df = generate_synthetic_sales_data()
        save_data(df, config['data']['raw_data_path'].split('/')[-1])
        print(f"Synthetic data generated and saved to {config['data']['raw_data_path']}")
    else:
        print("\n3. Raw data already exists, skipping generation...")
    
    # Preprocess data
    print("\n4. Preprocessing data...")
    processed_df = preprocess_data(
        input_filepath=config['data']['raw_data_path'],
        output_filepath=config['data']['processed_data_path']
    )
    
    # Initialize and train models
    print("\n5. Training models...")
    forecaster = SalesForecaster(model_dir=config['models']['model_dir'])
    
    # Train and evaluate models
    try:
        forecaster.train_xgboost(processed_df, config['features']['target_column'])
        print("XGBoost model trained successfully")
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
    
    try:
        forecaster.train_prophet(processed_df, config['features']['target_column'])
        print("Prophet model trained successfully")
    except Exception as e:
        print(f"Error training Prophet model: {e}")
    
    try:
        forecaster.train_lstm(processed_df, config['features']['target_column'])
        print("LSTM model trained successfully")
    except Exception as e:
        print(f"Error training LSTM model: {e}")
    
    # Save trained models
    print("\n6. Saving trained models...")
    forecaster.save_models()
    
    # Generate predictions
    print("\n7. Generating predictions...")
    predictor = SalesPredictor(model_dir=config['models']['model_dir'])
    predictions = predictor.ensemble_predict(
        processed_df['date'].max(),
        periods=30,
        weights=config['ensemble']['weights']
    )
    
    # Save predictions
    os.makedirs(os.path.dirname(config['data']['predictions_path']), exist_ok=True)
    predictions.to_csv(config['data']['predictions_path'], index=False)
    print(f"Predictions saved to {config['data']['predictions_path']}")
    
    # Generate visualizations
    print("\n8. Generating visualizations...")
    visualizer = SalesVisualizer(output_dir=config['visualization']['output_dir'])
    
    # Create various plots
    visualizer.plot_sales_trend(processed_df)
    visualizer.plot_seasonal_patterns(processed_df)
    visualizer.plot_product_performance(processed_df)
    visualizer.plot_regional_analysis(processed_df)
    visualizer.plot_forecast_comparison(processed_df, predictions)
    
    # Try to plot model performance if metrics exist
    try:
        metrics_path = os.path.join(config['models']['model_dir'], 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            visualizer.plot_model_performance(metrics)
    except Exception as e:
        print(f"Could not generate model performance visualization: {e}")
    
    # Calculate and print execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("\nPipeline completed successfully!")
    print(f"Total execution time: {execution_time}")
    print("\nResults can be found in:")
    print(f"- Raw data: {config['data']['raw_data_path']}")
    print(f"- Processed data: {config['data']['processed_data_path']}")
    print(f"- Predictions: {config['data']['predictions_path']}")
    print(f"- Models: {config['models']['model_dir']}")
    print(f"- Visualizations: {config['visualization']['output_dir']}")

if __name__ == '__main__':
    main() 