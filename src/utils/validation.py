from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    required_columns: List[str]
    date_column: str
    numeric_columns: List[str]
    categorical_columns: List[str]
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    
class DataValidator:
    """Validates data quality and consistency."""
    
    def __init__(self, config: DataValidationConfig):
        """Initialize the validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.validation_results: Dict[str, Any] = {}
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """Validate data types of columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        # Validate date column
        try:
            pd.to_datetime(df[self.config.date_column])
        except Exception as e:
            logger.error(f"Invalid date format in column {self.config.date_column}: {e}")
            return False
        
        # Validate numeric columns
        for col in self.config.numeric_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                logger.error(f"Column {col} should be numeric")
                return False
        
        return True
    
    def validate_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing value percentages.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of column names and their missing value percentages
        """
        missing_pcts = (df.isnull().sum() / len(df) * 100).to_dict()
        
        for col, pct in missing_pcts.items():
            if pct > 0:
                logger.warning(f"Column {col} has {pct:.2f}% missing values")
        
        return missing_pcts
    
    def validate_date_range(self, df: pd.DataFrame) -> bool:
        """Validate date range if specified in config.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes, False otherwise
        """
        dates = pd.to_datetime(df[self.config.date_column])
        
        if self.config.min_date:
            min_date = pd.to_datetime(self.config.min_date)
            if dates.min() < min_date:
                logger.error(f"Dates before {min_date} found")
                return False
        
        if self.config.max_date:
            max_date = pd.to_datetime(self.config.max_date)
            if dates.max() > max_date:
                logger.error(f"Dates after {max_date} found")
                return False
        
        return True
    
    def validate_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Number of duplicate rows
        """
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate rows")
        return n_duplicates
    
    def validate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {
            'schema_valid': self.validate_schema(df),
            'data_types_valid': self.validate_data_types(df),
            'missing_values': self.validate_missing_values(df),
            'date_range_valid': self.validate_date_range(df),
            'n_duplicates': self.validate_duplicates(df),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.validation_results
    
    def is_valid(self) -> bool:
        """Check if all validations passed.
        
        Returns:
            True if all validations passed, False otherwise
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_all() first.")
        
        return all([
            self.validation_results['schema_valid'],
            self.validation_results['data_types_valid'],
            self.validation_results['date_range_valid'],
            self.validation_results['n_duplicates'] == 0
        ]) 