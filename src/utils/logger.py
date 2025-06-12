import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
import time
from typing import Optional, Callable, Any


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)


def setup_logger(
    name: str = "facial-vision",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        console: Enable console output
        json_format: Use JSON format for logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create logs directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # File handler with rotation
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatters
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console:
        # Console uses regular format even if file uses JSON
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
            
    return wrapper


def performance_logger(operation: str) -> Callable:
    """Context manager for logging performance metrics"""
    class PerformanceLogger:
        def __init__(self, operation: str):
            self.operation = operation
            self.logger = get_logger("performance")
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            self.logger.debug(f"Starting operation: {self.operation}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(
                    f"Operation '{self.operation}' completed in {elapsed:.2f}s"
                )
            else:
                self.logger.error(
                    f"Operation '{self.operation}' failed after {elapsed:.2f}s: {exc_val}"
                )
                
    return PerformanceLogger(operation)


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter to add extra fields to log records"""
    
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['extra_fields'] = self.extra
        return msg, kwargs


# Initialize default logger
default_logger = setup_logger()


# Convenience functions
def debug(msg: str, **kwargs):
    """Log debug message"""
    default_logger.debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """Log info message"""
    default_logger.info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """Log warning message"""
    default_logger.warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """Log error message"""
    default_logger.error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """Log critical message"""
    default_logger.critical(msg, **kwargs)