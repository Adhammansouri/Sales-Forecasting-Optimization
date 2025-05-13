import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from src.data.generate_synthetic_data import generate_synthetic_sales_data, save_data
from src.visualization.visualize import SalesVisualizer

def main():
    """Simple demonstration of data generation and visualization."""
    print("Generating synthetic sales data...")
    df = generate_synthetic_sales_data()
    
    # Save data
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/predictions', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    save_data(df, 'sales_data.csv')
    print(f"Data saved to data/raw/sales_data.csv")
    
    # Parse date column as datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create visualizer
    visualizer = SalesVisualizer(output_dir='visualizations')
    
    print("Generating visualizations...")
    
    # Basic visualizations - skipping seasonal_patterns for now
    visualizer.plot_sales_trend(df)
    visualizer.plot_product_performance(df)
    visualizer.plot_regional_analysis(df)
    
    # Create a simple forecast for demonstration
    print("Generating sample forecast...")
    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq='D'
    )
    
    # Simple prediction (using average daily sales and adding some noise)
    avg_daily_sales = df.groupby('date')['units_sold'].sum().mean()
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'ensemble_prediction': [
            avg_daily_sales * (1 + 0.1 * np.sin(i) + 0.05 * np.random.randn())
            for i in range(len(future_dates))
        ]
    })
    
    # Save forecast
    forecast_df.to_csv('data/predictions/forecast_results.csv', index=False)
    print(f"Sample forecast saved to data/predictions/forecast_results.csv")
    
    # Plot forecast
    visualizer.plot_forecast_comparison(df, forecast_df)
    
    # Create mock model performance metrics
    metrics = {
        'xgboost': {'mae': 12.45, 'rmse': 15.67, 'r2': 0.92},
        'prophet': {'mae': 13.78, 'rmse': 16.89, 'r2': 0.89},
        'lstm': {'mae': 11.23, 'rmse': 14.56, 'r2': 0.94},
        'ensemble': {'mae': 10.12, 'rmse': 13.45, 'r2': 0.96}
    }
    
    # Plot model performance
    visualizer.plot_model_performance(metrics)
    
    print("\nDemo completed! Results saved to:")
    print("- Raw data: data/raw/sales_data.csv")
    print("- Forecast: data/predictions/forecast_results.csv")
    print("- Visualizations: visualizations/")
    
    return 0

if __name__ == '__main__':
    exit(main()) 