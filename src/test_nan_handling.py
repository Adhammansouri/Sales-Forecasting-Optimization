import logging
import pandas as pd
import numpy as np
from src.data.preprocess_data import preprocess_data, check_for_nan
from src.data.generate_synthetic_data import generate_synthetic_sales_data, save_data
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test NaN handling in the preprocessing pipeline."""
    # Create directories if needed
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Generate or load data
    input_filepath = 'data/raw/sales_data.csv'
    if not os.path.exists(input_filepath):
        logger.info("Generating synthetic data...")
        df = generate_synthetic_sales_data()
        save_data(df, 'sales_data.csv')
        logger.info(f"Synthetic data saved to {input_filepath}")
    else:
        logger.info(f"Loading existing data from {input_filepath}")
        df = pd.read_csv(input_filepath, parse_dates=['date'])
    
    # Step 2: Check for initial NaN values
    logger.info("Checking for NaN values in raw data...")
    has_nans = check_for_nan(df, "raw data")
    
    # Step 3: Introduce some NaN values if none exist (to test handling)
    if not has_nans:
        logger.info("Introducing some NaN values for testing...")
        # Create a mask for 2% of the data points
        mask = np.random.random(size=df.shape) < 0.02
        # Apply mask only to numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            col_idx = df.columns.get_loc(col)
            df.loc[mask[:, col_idx], col] = np.nan
        
        logger.info("After introducing NaN values:")
        check_for_nan(df, "artificially introduced NaNs")
        
        # Save this version with NaNs
        df.to_csv('data/raw/sales_data_with_nans.csv', index=False)
        logger.info("Saved data with NaNs to data/raw/sales_data_with_nans.csv")
    
    # Step 4: Process the data with different NaN handling strategies
    strategies = ['mean', 'median', 'zero']
    for strategy in strategies:
        logger.info(f"\n{'='*50}\nTesting NaN handling with strategy: {strategy}\n{'='*50}")
        processed_df = preprocess_data(
            input_filepath='data/raw/sales_data_with_nans.csv' if not has_nans else input_filepath,
            output_filepath=f'data/processed/processed_sales_data_{strategy}.csv',
            nan_strategy=strategy
        )
        
        # Check if any NaNs remain
        final_check = check_for_nan(processed_df, f"final ({strategy} strategy)")
        if not final_check:
            logger.info(f"SUCCESS: No NaNs remain after preprocessing with '{strategy}' strategy")
        else:
            logger.error(f"FAILURE: NaNs still present after preprocessing with '{strategy}' strategy")
    
    logger.info("\nTest completed! Check the log output to see which strategy worked best.")
    
if __name__ == '__main__':
    main() 