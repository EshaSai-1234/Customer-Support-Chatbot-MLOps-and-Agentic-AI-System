"""
Centralized Logging Utility for Agentic Customer Insight Platform
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "customer_insight", level: Optional[int] = None) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)

    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
