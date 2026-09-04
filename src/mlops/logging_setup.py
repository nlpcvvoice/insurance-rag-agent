import logging
import time
from pathlib import Path
from functools import wraps

LOGGER_NAME = "insurance_rag"

def setup_logging(level: str = "INFO", log_file: str = ""):
    """Configure root logging to console and an optional rolling file."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger():
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger = setup_logging()
    return logger


def timed(func):
    """Decorator that logs the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            get_logger().info(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception:
            get_logger().exception(f"Error in {func.__name__}")
            raise
    return wrapper
