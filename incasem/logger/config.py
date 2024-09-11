"""Defining a custom logging configuration
DOCS: https://docs.python.org/3/library/logging.config.html#user-defined-objects
Config from: https://madewithml.com/courses/mlops/logging/
"""

from loguru import logger
from pathlib import Path

logs_path = Path("logs")
logs_path.mkdir(exist_ok=True)
log_file_path = logs_path.joinpath("main.log")
logger.add(
    log_file_path, rotation="1 week", retention="10 days", backtrace=True, diagnose=True
)
error_log_file_path = logs_path.joinpath("error.log")
logger.add(
    error_log_file_path,
    level="ERROR",
    rotation="1 week",
    retention="10 days",
    backtrace=True,
    diagnose=True,
)
