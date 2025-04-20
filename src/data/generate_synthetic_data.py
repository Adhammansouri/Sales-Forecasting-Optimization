import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_sales_data(start_date='2020-01-01', num_days=1095):
    """
    Generate synthetic sales data with realistic patterns.
    
    Args:
        start_date (str): Start date for the data
        num_days (int): Number of days to generate data for
    
    Returns:
        pd.DataFrame: Synthetic sales data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    # Define products
    products = [
        'Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch',
        'Speaker', 'Camera', 'Monitor', 'Keyboard', 'Mouse'
    ]
    
    # Define regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Create base data
    data = []
    
    for date in dates:
        for product in products:
            for region in regions:
                # Base sales (random but consistent for each product)
                base_sales = np.random.normal(
                    loc=50 + hash(product) % 50,
                    scale=10
                )
                
                # Add seasonality
                season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Add weekly pattern
                weekly_factor = 1.0 + 0.2 * (1 if date.weekday() < 5 else -1)
                
                # Add trend
                trend_factor = 1.0 + 0.0003 * (date - pd.Timestamp(start_date)).days
                
                # Product-specific seasonality
                if product in ['Laptop', 'Tablet']:
                    # Higher sales during back-to-school season
                    if date.month in [8, 9]:
                        season_factor *= 1.5
                elif product in ['Smartphone', 'Smartwatch']:
                    # Higher sales during holiday season
                    if date.month == 12:
                        season_factor *= 2.0
                
                # Calculate final units sold
                units_sold = int(max(0, base_sales * season_factor * weekly_factor * trend_factor))
                
                # Generate price with small random variations
                base_price = {
                    'Laptop': 1200,
                    'Smartphone': 800,
                    'Tablet': 500,
                    'Headphones': 200,
                    'Smartwatch': 300,
                    'Speaker': 150,
                    'Camera': 600,
                    'Monitor': 400,
                    'Keyboard': 100,
                    'Mouse': 50
                }[product]
                
                price = base_price * np.random.normal(1, 0.05)  # 5% variation
                
                data.append({
                    'date': date,
                    'product': product,
                    'region': region,
                    'units_sold': units_sold,
                    'unit_price': round(price, 2),
                    'total_sales': round(units_sold * price, 2)
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some random promotions
    df['promotion'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def save_data(df, filename='sales_data.csv'):
    """Save the generated data to a CSV file."""
    output_path = 'data/raw/' + filename
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    # Generate three years of daily sales data
    df = generate_synthetic_sales_data()
    save_data(df)
    
    print("Sample of generated data:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nSummary statistics:")
    print(df.describe()) 