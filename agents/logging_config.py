"""
agents/logging_config.py — Centralized logging configuration for agents.

Provides a factory function to create properly configured loggers
with consistent formatting across all agent modules.
"""

import logging
import sys


def get_agent_logger(module_name: str) -> logging.Logger:
    """
    Create or retrieve a logger with consistent formatting.

    Args:
        module_name: The logger name, typically "travelmind.agents.<agent_name>"

    Returns:
        Configured logger instance with StreamHandler and standard formatter.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger
