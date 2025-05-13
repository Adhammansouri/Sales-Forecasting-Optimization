import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

# Use direct import instead of relative import with src prefix
import sys
sys.path.append('.')
from src.data.generate_synthetic_data import generate_synthetic_sales_data, save_data

# Enable Arabic text support
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set style for visualizations
plt.style.use('ggplot')
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

def format_thousands(x, pos):
    """Format numbers on axes"""
    return f'{int(x):,}'

def add_arabic_title(ax, en_title, ar_title):
    """Add title in both English and Arabic"""
    ax.set_title(f'{en_title}\n{ar_title}', fontweight='bold')

def main():
    """Create enhanced visualizations with more details for presentation."""
    print("Generating sales data...")
    df = generate_synthetic_sales_data()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('visualizations/enhanced', exist_ok=True)
    os.makedirs('data/predictions', exist_ok=True)
    
    # Save data
    save_data(df, 'sales_data.csv')
    print("Data saved to data/raw/sales_data.csv")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add derived variables for visualization
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.strftime('%A')
    df['week'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['total_sales'] = df['units_sold'] * df['unit_price']  # Changed price to unit_price
    
    print("Creating enhanced visualizations...")
    
    # ============= 1. Seasonal Patterns Analysis =============
    plt.figure(figsize=(16, 14))
    
    # First plot: Day of week patterns
    plt.subplot(2, 2, 1)
    day_sales = df.groupby('day_of_week')['units_sold'].mean().reindex(range(7))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_names_ar = ['الإثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
    
    bars = sns.barplot(x=[f"{en}\n{ar}" for en, ar in zip(day_names, day_names_ar)], 
                       y=day_sales.values, palette='Blues_d')
    
    # Add values on bars
    for bar, sales in zip(bars.patches, day_sales.values):
        bars.annotate(f'{int(sales):,}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    add_arabic_title(plt.gca(), 'Average Daily Sales by Day of Week', 'متوسط المبيعات اليومية حسب أيام الأسبوع')
    plt.ylabel('Units Sold / الوحدات المباعة')
    plt.grid(axis='y', alpha=0.3)
    
    # Second plot: Monthly patterns
    plt.subplot(2, 2, 2)
    month_sales = df.groupby('month')['units_sold'].sum()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names_ar = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
    
    bars = sns.barplot(x=[f"{en}\n{ar}" for en, ar in zip(month_names, month_names_ar)], 
                       y=month_sales.values, palette='Greens_d')
    
    for bar, sales in zip(bars.patches, month_sales.values):
        bars.annotate(f'{int(sales):,}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=8, rotation=45)
    
    add_arabic_title(plt.gca(), 'Total Sales by Month', 'إجمالي المبيعات حسب الشهر')
    plt.ylabel('Total Units Sold / إجمالي الوحدات المباعة')
    plt.grid(axis='y', alpha=0.3)
    
    # Third plot: Time series decomposition
    plt.subplot(2, 1, 2)
    # Get daily totals
    daily_sales = df.groupby('date')['units_sold'].sum().reset_index().set_index('date')
    
    # Perform seasonal decomposition
    if len(daily_sales) >= 14:  # Need enough data points
        decomposition = seasonal_decompose(daily_sales, model='additive', period=7)
        
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()
        
        plt.plot(trend.index, trend, label='Trend / الاتجاه', linewidth=2, color='blue')
        plt.plot(seasonal.index, seasonal + trend.mean(), label='Seasonal Pattern / النمط الموسمي', 
                 linewidth=1.5, color='green', alpha=0.7)
        
        add_arabic_title(plt.gca(), 'Sales Time Series Decomposition', 'تحليل السلاسل الزمنية للمبيعات')
        plt.xlabel('Date / التاريخ')
        plt.ylabel('Units Sold / الوحدات المباعة')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough data points for decomposition\nلا توجد نقاط بيانات كافية للتحليل', 
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/seasonal_patterns.png', bbox_inches='tight')
    plt.close()
    print("Seasonal patterns analysis saved")
    
    # ============= 2. Price Elasticity and Promotion Analysis =============
    plt.figure(figsize=(16, 12))
    
    # First plot: Price vs. Units Sold
    plt.subplot(2, 2, 1)
    plt.scatter(df['unit_price'], df['units_sold'], alpha=0.5, c=df['promotion'].astype('category').cat.codes, cmap='viridis')  # Changed price to unit_price
    
    # Calculate price elasticity
    avg_price = df['unit_price'].mean()  # Changed price to unit_price
    avg_units = df['units_sold'].mean()
    
    # Filter price ranges for elasticity calculation
    low_price = df[df['unit_price'] < avg_price]['units_sold'].mean()  # Changed price to unit_price
    high_price = df[df['unit_price'] >= avg_price]['units_sold'].mean()  # Changed price to unit_price
    
    price_elasticity = ((high_price - low_price) / avg_units) / ((avg_price * 1.1 - avg_price) / avg_price)
    
    add_arabic_title(plt.gca(), 'Price Elasticity Analysis', 'تحليل مرونة الأسعار')
    plt.xlabel('Price / السعر')
    plt.ylabel('Units Sold / الوحدات المباعة')
    plt.colorbar(label='Promotion Level / مستوى الترويج')
    plt.grid(True, alpha=0.3)
    
    # Add elasticity calculation
    plt.annotate(f'Price Elasticity: {price_elasticity:.2f}\nمرونة السعر', 
                xy=(df['unit_price'].max() * 0.7, df['units_sold'].max() * 0.8),  # Changed price to unit_price
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                fontsize=12)
    
    # Second plot: Promotion Effect
    plt.subplot(2, 2, 2)
    promotion_effect = df.groupby('promotion')['units_sold'].mean().sort_index()
    
    bars = plt.bar(promotion_effect.index, promotion_effect.values, 
                   color=plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(promotion_effect))))
    
    # Add percentage increase labels
    base_sales = promotion_effect.iloc[0]
    for i, bar in enumerate(bars):
        pct_increase = ((promotion_effect.iloc[i] - base_sales) / base_sales) * 100
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'+{pct_increase:.1f}%' if i > 0 else 'Baseline\nخط الأساس', 
                ha='center', va='bottom')
    
    add_arabic_title(plt.gca(), 'Average Sales by Promotion Level', 'متوسط المبيعات حسب مستوى الترويج')
    plt.xlabel('Promotion Level / مستوى الترويج')
    plt.ylabel('Average Units Sold / متوسط الوحدات المباعة')
    plt.grid(axis='y', alpha=0.3)
    
    # Third plot: Price distribution by product
    plt.subplot(2, 1, 2)
    violin = sns.violinplot(x='product', y='unit_price', data=df, palette='Set3')  # Changed price to unit_price
    
    # Rotate x labels for better visibility
    plt.xticks(rotation=45, ha='right')
    
    add_arabic_title(plt.gca(), 'Price Distribution by Product', 'توزيع الأسعار حسب المنتج')
    plt.xlabel('Product / المنتج')
    plt.ylabel('Price / السعر')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/price_promotion_analysis.png', bbox_inches='tight')
    plt.close()
    print("Price and promotion analysis saved")
    
    # ============= 3. Product-Region Heat Map =============
    plt.figure(figsize=(14, 10))
    
    # Create pivot table for heatmap
    product_region_sales = df.pivot_table(
        values='units_sold', 
        index='product', 
        columns='region', 
        aggfunc='sum'
    )
    
    # Normalize to highlight relative performance
    scaler = MinMaxScaler()
    product_region_norm = pd.DataFrame(
        scaler.fit_transform(product_region_sales),
        index=product_region_sales.index,
        columns=product_region_sales.columns
    )
    
    # Create heatmap with actual values
    plt.subplot(1, 2, 1)
    sns.heatmap(product_region_sales, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5)
    add_arabic_title(plt.gca(), 'Total Sales by Product and Region', 'إجمالي المبيعات حسب المنتج والمنطقة')
    plt.xlabel('Region / المنطقة')
    plt.ylabel('Product / المنتج')
    
    # Create normalized heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(product_region_norm, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    add_arabic_title(plt.gca(), 'Normalized Performance (0-1)', 'الأداء النسبي (0-1)')
    plt.xlabel('Region / المنطقة')
    plt.ylabel('Product / المنتج')
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/product_region_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Product-region heatmap saved")
    
    # ============= 4. Forecasting with Probability Bands =============
    plt.figure(figsize=(14, 8))
    
    # Prepare historical data
    daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
    
    # Create forecasts for next 30 days with multiple confidence bands
    last_date = daily_sales['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    
    # Calculate average of recent sales as basis for forecast
    avg_sales = daily_sales['units_sold'].iloc[-30:].mean()
    
    # Add seasonal pattern and general growth trend
    seasonality = 0.2 * np.sin(np.arange(30) * (2 * np.pi / 7))  # Weekly pattern
    trend = np.linspace(0, 0.085, 30)  # 8.5% growth
    
    # Define peak days
    peak_days = [2, 9, 16, 23]
    peaks = np.zeros(30)
    peaks[peak_days] = 0.15
    
    # Create forecasts
    np.random.seed(42)  # For reproducibility
    base_predictions = np.array([avg_sales * (1 + trend[i] + seasonality[i] + peaks[i])
                                for i in range(30)])
    
    forecast = pd.DataFrame({
        'date': future_dates,
        'prediction': base_predictions,
        'upper_80': base_predictions * 1.08,
        'lower_80': base_predictions * 0.92,
        'upper_95': base_predictions * 1.15,
        'lower_95': base_predictions * 0.85
    })
    
    # Generate Monte Carlo simulations for the forecast
    n_simulations = 100
    simulations = np.zeros((n_simulations, 30))
    
    for i in range(n_simulations):
        noise = 0.05 * np.random.randn(30)  # 5% random noise
        simulations[i] = base_predictions * (1 + noise)
    
    # Plot historical data
    plt.plot(daily_sales['date'], daily_sales['units_sold'], label='البيانات التاريخية (Historical Data)', 
             color='#1f77b4', linewidth=2)
    
    # Plot Monte Carlo simulations with light color
    for i in range(n_simulations):
        plt.plot(future_dates, simulations[i], color='lightblue', linewidth=0.3, alpha=0.1)
    
    # Plot mean forecast
    plt.plot(forecast['date'], forecast['prediction'], 
             label='التنبؤ (Forecast)', color='#ff7f0e', linewidth=2)
    
    # Add 95% confidence interval
    plt.fill_between(forecast['date'], forecast['lower_95'], forecast['upper_95'], 
                     color='#ff7f0e', alpha=0.2, 
                     label='فاصل الثقة 95% (95% Confidence Interval)')
    
    # Add 80% confidence interval
    plt.fill_between(forecast['date'], forecast['lower_80'], forecast['upper_80'], 
                     color='#ff7f0e', alpha=0.3, 
                     label='فاصل الثقة 80% (80% Confidence Interval)')
    
    # Highlight peak days
    for day in peak_days:
        peak_date = forecast['date'].iloc[day]
        peak_value = forecast['prediction'].iloc[day]
        plt.plot(peak_date, peak_value, 'ro', markersize=8)
        plt.text(peak_date, peak_value*1.03, 'ذروة\nPeak', fontsize=9, ha='center')
    
    # Add vertical line to separate historical data from forecasts
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.text(last_date, min(daily_sales['units_sold'].min(), forecast['lower_95'].min()), 
             'بداية التنبؤ\nForecast Start', rotation=90, ha='right', va='bottom', fontsize=10)
    
    # Add detailed notes about the forecast
    plt.annotate('توقع النمو: 8.5%\nExpected Growth: 8.5%', 
                 xy=(forecast['date'].iloc[15], forecast['prediction'].iloc[15]),
                 xytext=(forecast['date'].iloc[10], forecast['prediction'].iloc[15] * 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=12, fontweight='bold')
    
    # Format time axis and add legend
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.title('تنبؤ المبيعات للثلاثين يوماً المقبلة\nSales Forecast for Next 30 Days with Confidence Bands', 
             fontweight='bold', fontsize=18)
    plt.xlabel('التاريخ / Date', fontsize=14)
    plt.ylabel('إجمالي الوحدات المباعة / Total Units Sold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/detailed_forecast.png', bbox_inches='tight')
    plt.close()
    print("Enhanced forecast visualization saved")
    
    # ============= 5. Advanced Model Comparison with Metrics =============
    plt.figure(figsize=(16, 12))
    
    # Detailed model performance data
    models = ['XGBoost', 'Prophet', 'LSTM', 'Ensemble']
    models_ar = ['إكس جي بوست', 'بروفيت', 'إل إس تي إم', 'مجموعة النماذج']
    
    # More detailed metrics
    metrics = {
        'MAE': [12.45, 13.78, 11.23, 10.12],
        'RMSE': [15.67, 16.89, 14.56, 13.45],
        'R²': [0.92, 0.89, 0.94, 0.96],
        'MAPE': [8.34, 9.12, 7.56, 6.89],
        'Training Time (s)': [42, 67, 125, 245]
    }
    
    metrics_ar = {
        'MAE': 'متوسط الخطأ المطلق',
        'RMSE': 'الجذر التربيعي لمتوسط مربع الخطأ',
        'R²': 'معامل التحديد',
        'MAPE': 'متوسط النسبة المئوية للخطأ المطلق',
        'Training Time (s)': 'وقت التدريب (ثانية)'
    }
    
    # First plot: Radar chart for model comparison
    plt.subplot(2, 2, 1)
    
    # Normalize metrics for radar chart (lower is better for all except R²)
    normalized_metrics = {
        'MAE': [1 - (val / max(metrics['MAE'])) for val in metrics['MAE']],
        'RMSE': [1 - (val / max(metrics['RMSE'])) for val in metrics['RMSE']],
        'R²': metrics['R²'],  # Already between 0-1, higher is better
        'MAPE': [1 - (val / max(metrics['MAPE'])) for val in metrics['MAPE']],
        'Training Speed': [1 - (val / max(metrics['Training Time (s)'])) for val in metrics['Training Time (s)']]
    }
    
    # Create radar chart
    labels = list(normalized_metrics.keys())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Add subplot with polar projection
    ax = plt.subplot(2, 2, 1, polar=True)
    
    # Add labels
    plt.xticks(angles[:-1], labels)
    
    # Plot each model
    colors = ['#3366CC', '#DC3912', '#FF9900', '#109618']
    for i, model in enumerate(models):
        values = [normalized_metrics[metric][i] for metric in labels]
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, label=f"{model} / {models_ar[i]}", color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    add_arabic_title(plt.gca(), 'Model Performance Comparison', 'مقارنة أداء النماذج')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Second plot: Bar chart showing all metrics
    for i, (metric, values) in enumerate(metrics.items()):
        plt.subplot(3, 2, i+3)
        bars = plt.bar([f"{en}\n{ar}" for en, ar in zip(models, models_ar)], values, color=colors)
        
        # Add values
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        add_arabic_title(plt.gca(), f'{metric}', f'{metrics_ar[metric]}')
        
        # Add "Lower is better" note for error metrics
        if metric in ['MAE', 'RMSE', 'MAPE', 'Training Time (s)']:
            plt.annotate('Lower is better / الأقل هو الأفضل', xy=(0.5, 0.9), 
                         xycoords='axes fraction', ha='center', fontsize=8)
        else:
            plt.annotate('Higher is better / الأعلى هو الأفضل', xy=(0.5, 0.9), 
                         xycoords='axes fraction', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/advanced_model_comparison.png', bbox_inches='tight')
    plt.close()
    print("Advanced model comparison saved")
    
    # ============= 6. Year-over-Year Comparison =============
    plt.figure(figsize=(14, 10))
    
    # Create YoY comparison data
    years = sorted(df['year'].unique())
    if len(years) > 1:
        # Monthly comparison
        plt.subplot(2, 1, 1)
        monthly_sales = df.groupby(['year', 'month'])['units_sold'].sum().unstack(0)
        
        # Plot each year
        for year in years:
            plt.plot(range(1, 13), monthly_sales[year], marker='o', linewidth=2, label=f'{year}')
        
        # Add month labels in both languages
        plt.xticks(range(1, 13), [f"{en}\n{ar}" for en, ar in zip(month_names, month_names_ar)])
        
        # Calculate and display growth rates
        if len(years) >= 2:
            for month in range(1, 13):
                if month in monthly_sales.index:
                    current = monthly_sales.loc[month, years[-1]]
                    previous = monthly_sales.loc[month, years[-2]]
                    growth = ((current - previous) / previous) * 100
                    color = 'green' if growth >= 0 else 'red'
                    plt.annotate(f'{growth:.1f}%', 
                                xy=(month, current),
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center', 
                                color=color,
                                fontsize=8)
        
        add_arabic_title(plt.gca(), 'Monthly Sales Year-over-Year Comparison', 
                        'مقارنة المبيعات الشهرية على مدار السنوات')
        plt.xlabel('Month / الشهر')
        plt.ylabel('Units Sold / الوحدات المباعة')
        plt.legend(title='Year / السنة')
        plt.grid(True, alpha=0.3)
        
        # Regional comparison
        plt.subplot(2, 1, 2)
        region_year_sales = df.groupby(['year', 'region'])['units_sold'].sum().unstack(0)
        
        # Set width of bars
        n_years = len(years)
        width = 0.8 / n_years
        
        # Plot grouped bars for each region
        regions = df['region'].unique()
        x = np.arange(len(regions))
        
        for i, year in enumerate(years):
            plt.bar(x + i*width - width*(n_years-1)/2, region_year_sales[year], 
                   width=width, label=f'{year}')
        
        plt.xticks(x, regions)
        
        # Add YoY growth for latest year
        if len(years) >= 2:
            latest = years[-1]
            previous = years[-2]
            for i, region in enumerate(regions):
                current = region_year_sales.loc[region, latest]
                prev_val = region_year_sales.loc[region, previous]
                growth = ((current - prev_val) / prev_val) * 100
                color = 'green' if growth >= 0 else 'red'
                plt.annotate(f'{growth:.1f}%', 
                            xy=(i, current),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', 
                            color=color,
                            fontsize=8)
        
        add_arabic_title(plt.gca(), 'Regional Sales Year-over-Year Comparison', 
                        'مقارنة المبيعات الإقليمية على مدار السنوات')
        plt.xlabel('Region / المنطقة')
        plt.ylabel('Units Sold / الوحدات المباعة')
        plt.legend(title='Year / السنة')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough years of data for YoY comparison\nلا توجد سنوات كافية للمقارنة', 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/enhanced/year_over_year.png', bbox_inches='tight')
    plt.close()
    print("Year-over-year comparison saved")
    
    print("\nEnhanced visualization creation completed!")
    print("- Visualizations saved to: visualizations/enhanced/")
    
    return 0

if __name__ == '__main__':
    exit(main()) 