import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class SalesVisualizer:
    """Class for generating visualizations of sales data and forecasts."""
    
    def __init__(self, output_dir='visualizations', style='ggplot', dpi=300):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style to use
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use(style)
    
    def plot_sales_trend(self, df):
        """Plot overall sales trend over time.
        
        Args:
            df: DataFrame with processed data
        """
        plt.figure(figsize=(12, 6))
        
        # Group by date and sum sales
        daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
        
        # Plot
        plt.plot(daily_sales['date'], daily_sales['units_sold'], lw=2)
        
        # Add trend line
        z = np.polyfit(np.arange(len(daily_sales)), daily_sales['units_sold'], 1)
        p = np.poly1d(z)
        plt.plot(daily_sales['date'], p(np.arange(len(daily_sales))), "r--", lw=1, alpha=0.7)
        
        # Customize
        plt.title('Overall Sales Trend', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Units Sold', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'sales_trend.png'), dpi=self.dpi)
        plt.close()
        
        print("Sales trend plot saved")
    
    def plot_seasonal_patterns(self, df):
        """Plot seasonal patterns in sales.
        
        Args:
            df: DataFrame with processed data
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Monthly seasonality
        monthly_sales = df.groupby([df['date'].dt.year, df['date'].dt.month])['units_sold'].sum().reset_index()
        monthly_sales.columns = ['Year', 'Month', 'Total Sales']
        monthly_sales_pivot = monthly_sales.pivot(index='Month', columns='Year', values='Total Sales')
        
        monthly_sales_pivot.plot(ax=axes[0])
        axes[0].set_title('Monthly Sales by Year', fontsize=14)
        axes[0].set_xlabel('Month', fontsize=12)
        axes[0].set_ylabel('Total Units Sold', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Weekly seasonality
        weekly_sales = df.groupby(df['date'].dt.dayofweek)['units_sold'].mean().reset_index()
        weekly_sales.columns = ['Day of Week', 'Average Units Sold']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_sales['Day Name'] = weekly_sales['Day of Week'].apply(lambda x: days[x])
        
        sns.barplot(x='Day Name', y='Average Units Sold', data=weekly_sales, ax=axes[1])
        axes[1].set_title('Average Sales by Day of Week', fontsize=14)
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Average Units Sold', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'seasonal_patterns.png'), dpi=self.dpi)
        plt.close()
        
        print("Seasonal patterns plot saved")
    
    def plot_product_performance(self, df):
        """Plot product performance comparison.
        
        Args:
            df: DataFrame with processed data
        """
        plt.figure(figsize=(12, 8))
        
        # Group by product and sum sales
        product_sales = df.groupby('product')['units_sold'].sum().sort_values(ascending=False).reset_index()
        
        # Plot
        ax = sns.barplot(x='product', y='units_sold', data=product_sales)
        
        # Customize
        plt.title('Total Sales by Product', fontsize=16)
        plt.xlabel('Product', fontsize=12)
        plt.ylabel('Total Units Sold', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'product_performance.png'), dpi=self.dpi)
        plt.close()
        
        print("Product performance plot saved")
    
    def plot_regional_analysis(self, df):
        """Plot regional sales analysis.
        
        Args:
            df: DataFrame with processed data
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Regional total sales
        region_sales = df.groupby('region')['units_sold'].sum().sort_values(ascending=False).reset_index()
        sns.barplot(x='region', y='units_sold', data=region_sales, ax=axes[0])
        axes[0].set_title('Total Sales by Region', fontsize=14)
        axes[0].set_xlabel('Region', fontsize=12)
        axes[0].set_ylabel('Total Units Sold', fontsize=12)
        
        # Regional trend over time
        region_time_sales = df.groupby(['date', 'region'])['units_sold'].sum().reset_index()
        for region in region_time_sales['region'].unique():
            subset = region_time_sales[region_time_sales['region'] == region]
            axes[1].plot(subset['date'], subset['units_sold'], label=region)
        
        axes[1].set_title('Sales Trends by Region', fontsize=14)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Total Units Sold', fontsize=12)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regional_analysis.png'), dpi=self.dpi)
        plt.close()
        
        print("Regional analysis plot saved")
    
    def plot_forecast_comparison(self, historical_df, forecast_df):
        """Plot comparison of forecast with historical data.
        
        Args:
            historical_df: DataFrame with historical data
            forecast_df: DataFrame with forecasted data
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare historical data
        historical = historical_df.groupby('date')['units_sold'].sum().reset_index()
        historical['data_type'] = 'Historical'
        
        # Prepare forecast data - assuming 'ensemble_prediction' is the main forecast column
        forecast = forecast_df[['date', 'ensemble_prediction']].copy()
        forecast.columns = ['date', 'units_sold']
        forecast['data_type'] = 'Forecast'
        
        # Combine data
        combined = pd.concat([historical, forecast])
        
        # Plot
        sns.lineplot(x='date', y='units_sold', hue='data_type', data=combined, 
                    style='data_type', markers=True, dashes=False)
        
        # Customize
        plt.title('Historical Data vs. Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Units Sold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis date ticks
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'forecast_comparison.png'), dpi=self.dpi)
        plt.close()
        
        print("Forecast comparison plot saved")
    
    def plot_model_performance(self, metrics):
        """Plot model performance comparison.
        
        Args:
            metrics: Dictionary with model metrics
        """
        # Create DataFrame for easier plotting
        models = []
        maes = []
        rmses = []
        r2s = []
        
        for model_name, model_metrics in metrics.items():
            models.append(model_name)
            maes.append(model_metrics['mae'])
            rmses.append(model_metrics['rmse'])
            r2s.append(model_metrics['r2'])
        
        df = pd.DataFrame({
            'Model': models,
            'MAE': maes,
            'RMSE': rmses,
            'R²': r2s
        })
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MAE (lower is better)
        sns.barplot(x='Model', y='MAE', data=df, ax=axes[0])
        axes[0].set_title('Mean Absolute Error (Lower is Better)', fontsize=14)
        
        # RMSE (lower is better)
        sns.barplot(x='Model', y='RMSE', data=df, ax=axes[1])
        axes[1].set_title('Root Mean Squared Error (Lower is Better)', fontsize=14)
        
        # R² (higher is better)
        sns.barplot(x='Model', y='R²', data=df, ax=axes[2])
        axes[2].set_title('R² Score (Higher is Better)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_performance.png'), dpi=self.dpi)
        plt.close()
        
        print("Model performance plot saved")

if __name__ == '__main__':
    # Example usage
    from src.data.preprocess_data import preprocess_data
    
    # Get processed data
    df = pd.read_csv('data/processed/processed_sales_data.csv', parse_dates=['date'])
    
    # Create visualizer
    visualizer = SalesVisualizer()
    
    # Generate visualizations
    visualizer.plot_sales_trend(df)
    visualizer.plot_seasonal_patterns(df)
    visualizer.plot_product_performance(df)
    visualizer.plot_regional_analysis(df)
