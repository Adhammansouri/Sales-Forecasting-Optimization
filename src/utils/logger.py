import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file. If None, logs only to console
        level: Logging level
        format: Log message format
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters and handlers
    formatter = logging.Formatter(format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(
    name: str = __name__,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """Get a logger instance with optional file logging.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Optional directory for log files
        
    Returns:
        Configured logger instance
    """
    if log_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        return setup_logger(name, str(log_file))
    return setup_logger(name)

# Create a default logger for the application
app_logger = get_logger('sales_forecasting') 