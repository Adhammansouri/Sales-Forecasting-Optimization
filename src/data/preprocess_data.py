import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

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

def check_for_nan(df, stage_name="Unknown"):
    """Check for NaN values in the DataFrame and log their locations.
    
    Args:
        df: DataFrame to check
        stage_name: Name of the preprocessing stage (for logging)
        
    Returns:
        bool: True if NaN values were found, False otherwise
    """
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    
    if len(nan_columns) > 0:
        logger.warning(f"NaN values found after {stage_name} stage:")
        for col, count in nan_columns.items():
            logger.warning(f"  - {col}: {count} NaNs ({(count/len(df))*100:.2f}% of data)")
        return True
    return False

def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame to process
        strategy: Strategy to handle NaNs ('mean', 'median', 'zero', or 'drop')
        
    Returns:
        DataFrame with NaNs handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        # Drop rows with NaN values
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        logger.info(f"Dropped {dropped} rows with NaN values ({(dropped/original_len)*100:.2f}% of data)")
        return df
    
    # For numeric columns, fill NaNs based on strategy
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            if strategy == 'mean':
                fill_value = df[col].mean()
                logger.info(f"Filling NaNs in '{col}' with mean: {fill_value:.4f}")
            elif strategy == 'median':
                fill_value = df[col].median()
                logger.info(f"Filling NaNs in '{col}' with median: {fill_value:.4f}")
            elif strategy == 'zero':
                fill_value = 0
                logger.info(f"Filling NaNs in '{col}' with zero")
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            df[col] = df[col].fillna(fill_value)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_value = df[col].mode()[0]
            logger.info(f"Filling NaNs in '{col}' with mode: {mode_value}")
            df[col] = df[col].fillna(mode_value)
    
    return df

def preprocess_data(input_filepath='data/raw/sales_data.csv',
                   output_filepath='data/processed/processed_sales_data.csv',
                   nan_strategy='mean'):
    """Main function to preprocess the data."""
    # Load data
    df = load_data(input_filepath)
    logger.info(f"Loaded data from {input_filepath} with shape {df.shape}")
    
    # Check for NaNs in raw data
    check_for_nan(df, "initial load")
    
    # Add features
    df = add_time_features(df)
    check_for_nan(df, "adding time features")
    
    df = add_lag_features(df)
    check_for_nan(df, "adding lag features")
    
    # Handle missing values from lag features
    df = handle_missing_values(df, strategy=nan_strategy)
    
    df = add_rolling_features(df)
    check_for_nan(df, "adding rolling features")
    
    # Handle any missing values from rolling features
    df = handle_missing_values(df, strategy=nan_strategy)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    check_for_nan(df, "encoding categorical features")
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df)
    check_for_nan(df, "scaling numerical features")
    
    # Final check for NaNs
    has_nans = check_for_nan(df, "final check")
    if has_nans:
        logger.warning("WARNING: Processed data still contains NaN values. This may cause model training to fail.")
        # Final attempt to clean any remaining NaNs
        df = df.fillna(0)
        logger.info("Filled remaining NaNs with zeros as a last resort")
    
    # Save processed data
    df.to_csv(output_filepath, index=False)
    logger.info(f"Processed data saved to {output_filepath} with shape {df.shape}")
    
    return df

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process the data
    df = preprocess_data()
    
    print("\nSample of processed data:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nFeatures created:", df.columns.tolist()) 