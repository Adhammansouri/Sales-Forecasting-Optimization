import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def load_data(filepath='data/raw/sales_data.csv'):
    """Load the sales data from CSV file."""
    return pd.read_csv(filepath, parse_dates=['date'])

def add_time_features(df):
    """Add time-based features to the dataset."""
    df = df.copy()
    
    # Extract time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Create seasonal features
    df['season'] = pd.cut(df['month'],
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    return df

def add_lag_features(df, group_cols=['product', 'region'], target_col='units_sold', lags=[1, 7, 30]):
    """Add lagged features for the target variable."""
    df = df.copy()
    
    for lag in lags:
        for col in group_cols:
            df[f'{target_col}_lag_{lag}_{col}'] = df.groupby(col)[target_col].shift(lag)
    
    return df

def add_rolling_features(df, group_cols=['product', 'region'], target_col='units_sold',
                        windows=[7, 30, 90]):
    """Add rolling mean and std features."""
    df = df.copy()
    
    for window in windows:
        for col in group_cols:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}_{col}'] = (
                df.groupby(col)[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}_{col}'] = (
                df.groupby(col)[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using one-hot encoding."""
    df = df.copy()
    
    # One-hot encode categorical columns
    categorical_cols = ['product', 'region', 'season']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    return df

def scale_numerical_features(df, exclude_cols=['date']):
    """Scale numerical features using StandardScaler."""
    df = df.copy()
    
    # Identify numerical columns to scale
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

def prepare_data_for_training(df, target_col='units_sold', test_size=0.2):
    """Prepare data for training by splitting into features and target."""
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Remove target column and date from features
    feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def preprocess_data(input_filepath='data/raw/sales_data.csv',
                   output_filepath='data/processed/processed_sales_data.csv'):
    """Main function to preprocess the data."""
    # Load data
    df = load_data(input_filepath)
    
    # Add features
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df)
    
    # Save processed data
    df.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")
    
    return df

if __name__ == '__main__':
    # Process the data
    df = preprocess_data()
    
    print("\nSample of processed data:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nFeatures created:", df.columns.tolist()) 