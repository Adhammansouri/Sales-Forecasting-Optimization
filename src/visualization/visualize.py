import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

class SalesVisualizer:
    def __init__(self, output_dir='reports/figures'):
        """Initialize the visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn')
        
    def plot_sales_trend(self, df, save=True):
        """Plot overall sales trend."""
        fig = px.line(
            df,
            x='date',
            y='total_sales',
            title='Overall Sales Trend Over Time'
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'sales_trend.html'))
        return fig
    
    def plot_seasonal_patterns(self, df, save=True):
        """Plot seasonal patterns in sales."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Sales Pattern',
                'Day of Week Pattern',
                'Yearly Sales Pattern',
                'Weekly Sales Pattern'
            )
        )
        
        # Monthly pattern
        monthly_sales = df.groupby(df['date'].dt.month)['total_sales'].mean()
        fig.add_trace(
            go.Scatter(x=monthly_sales.index, y=monthly_sales.values, name='Monthly'),
            row=1, col=1
        )
        
        # Day of week pattern
        daily_sales = df.groupby(df['date'].dt.dayofweek)['total_sales'].mean()
        fig.add_trace(
            go.Scatter(x=daily_sales.index, y=daily_sales.values, name='Daily'),
            row=1, col=2
        )
        
        # Yearly pattern
        yearly_sales = df.groupby(df['date'].dt.year)['total_sales'].mean()
        fig.add_trace(
            go.Scatter(x=yearly_sales.index, y=yearly_sales.values, name='Yearly'),
            row=2, col=1
        )
        
        # Weekly pattern
        weekly_sales = df.groupby(df['date'].dt.isocalendar().week)['total_sales'].mean()
        fig.add_trace(
            go.Scatter(x=weekly_sales.index, y=weekly_sales.values, name='Weekly'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Seasonal Sales Patterns")
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'seasonal_patterns.html'))
        return fig
    
    def plot_product_performance(self, df, save=True):
        """Plot product-wise sales performance."""
        product_sales = df.groupby('product')['total_sales'].agg(['sum', 'mean'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Sales by Product', 'Average Sales by Product')
        )
        
        fig.add_trace(
            go.Bar(
                x=product_sales.index,
                y=product_sales['sum'],
                name='Total Sales'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=product_sales.index,
                y=product_sales['mean'],
                name='Average Sales'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Product Performance Analysis")
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'product_performance.html'))
        return fig
    
    def plot_regional_analysis(self, df, save=True):
        """Plot regional sales analysis."""
        regional_sales = df.groupby(['region', 'product'])['total_sales'].sum().unstack()
        
        fig = px.imshow(
            regional_sales,
            title='Regional Sales Heatmap by Product',
            labels=dict(x='Product', y='Region', color='Total Sales'),
            aspect='auto'
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'regional_analysis.html'))
        return fig
    
    def plot_forecast_comparison(self, actual, predictions, save=True):
        """Plot comparison of actual vs predicted values."""
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(
            go.Scatter(
                x=actual['date'],
                y=actual['total_sales'],
                name='Actual',
                line=dict(color='blue')
            )
        )
        
        # Plot predictions from each model
        colors = {'xgboost': 'red', 'prophet': 'green', 'lstm': 'orange'}
        for model in ['xgboost', 'prophet', 'lstm']:
            if f'{model}_prediction' in predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions['date'],
                        y=predictions[f'{model}_prediction'],
                        name=f'{model.upper()} Prediction',
                        line=dict(color=colors[model])
                    )
                )
        
        # Plot ensemble prediction
        fig.add_trace(
            go.Scatter(
                x=predictions['date'],
                y=predictions['ensemble_prediction'],
                name='Ensemble Prediction',
                line=dict(color='purple', width=2)
            )
        )
        
        fig.update_layout(
            title='Sales Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'forecast_comparison.html'))
        return fig
    
    def plot_model_performance(self, metrics, save=True):
        """Plot model performance metrics comparison."""
        models = list(metrics.keys())
        metrics_df = pd.DataFrame(metrics).T
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('MAE Comparison', 'RMSE Comparison', 'R² Score Comparison')
        )
        
        # Plot MAE
        fig.add_trace(
            go.Bar(x=models, y=metrics_df['mae'], name='MAE'),
            row=1, col=1
        )
        
        # Plot RMSE
        fig.add_trace(
            go.Bar(x=models, y=metrics_df['rmse'], name='RMSE'),
            row=1, col=2
        )
        
        # Plot R2
        fig.add_trace(
            go.Bar(x=models, y=metrics_df['r2'], name='R²'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, title_text="Model Performance Comparison")
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'model_performance.html'))
        return fig

def main():
    """Main function to generate all visualizations."""
    # Load data
    raw_data = pd.read_csv('data/raw/sales_data.csv', parse_dates=['date'])
    processed_data = pd.read_csv('data/processed/processed_sales_data.csv', parse_dates=['date'])
    predictions = pd.read_csv('data/predictions/forecast.csv', parse_dates=['date'])
    
    # Initialize visualizer
    visualizer = SalesVisualizer()
    
    # Generate visualizations
    visualizer.plot_sales_trend(raw_data)
    visualizer.plot_seasonal_patterns(raw_data)
    visualizer.plot_product_performance(raw_data)
    visualizer.plot_regional_analysis(raw_data)
    visualizer.plot_forecast_comparison(raw_data, predictions)
    
    # Load and plot model performance
    try:
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        visualizer.plot_model_performance(metrics)
    except Exception as e:
        print(f"Could not plot model performance: {e}")
    
    print("All visualizations have been generated and saved in the reports/figures directory.")

if __name__ == '__main__':
    main() 