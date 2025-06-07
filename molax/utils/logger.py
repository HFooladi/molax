import datetime
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "molax",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Set up a logger with both file and console handlers.

    This function configures a logger with the following features:
    - Timestamp in log messages
    - Log level (INFO, DEBUG, etc.)
    - Both file and console output (configurable)
    - Detailed formatting including module name and line number
    - Automatic log file rotation

    Args:
        name: Name of the logger. Defaults to "molax"
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file. If None, no file logging is performed
        console_output: Whether to output logs to console. Defaults to True

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Add file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add file handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_default_logger() -> logging.Logger:
    """Get the default logger for the molax package.

    This function creates a logger with default settings:
    - Logs to both console and file
    - File logging in 'logs/molax_YYYYMMDD.log'
    - INFO level logging
    - Includes timestamps and module information

    Returns:
        logging.Logger: Default logger instance
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"molax_{date_str}.log"

    return setup_logger(
        name="molax",
        log_level=logging.INFO,
        log_file=str(log_file),
        console_output=True,
    )


# Create default logger instance
logger = get_default_logger()
