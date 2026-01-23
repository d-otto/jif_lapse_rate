"""
Logging utilities for the JIFLR data pipeline.

Provides standardized logging configuration and formatting functions
for consistent output across all pipeline scripts.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Constants for formatting
SEPARATOR_WIDTH = 60
SEPARATOR_CHAR = "="
SUBSEPARATOR_CHAR = "-"
INDENT = "  "  # 2 spaces

# Environment variable for log file path
LOG_PATH_ENV_VAR = "JIFLR_PIPELINE_LOG"


class TqdmCompatibleHandler(logging.StreamHandler):
    """
    A logging handler that's compatible with tqdm progress bars.

    Uses tqdm.write() when tqdm is actively displaying progress bars,
    falls back to standard stream writing otherwise.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg, file=self.stream)
            except ImportError:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_pipeline_logging(
    log_file: Optional[Path] = None,
    step_number: Optional[int] = None,
    total_steps: Optional[int] = None,
    mode: str = "a",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure dual output logging (terminal + file) for pipeline scripts.

    Parameters
    ----------
    log_file : Path, optional
        Path to log file. If None, checks JIFLR_PIPELINE_LOG env var.
        If neither is set, logs only to terminal.
    step_number : int, optional
        Current step number (1-indexed) for context
    total_steps : int, optional
        Total number of steps for context
    mode : str
        File mode - "w" to overwrite (orchestrator), "a" to append (scripts)
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Get or create the pipeline logger
    logger = logging.getLogger("jiflr.pipeline")

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)

    # Create formatter (no timestamp prefix - we format output ourselves)
    formatter = logging.Formatter("%(message)s")

    # Add console handler (tqdm-compatible)
    console_handler = TqdmCompatibleHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Determine log file path
    if log_file is None:
        log_file_str = os.environ.get(LOG_PATH_ENV_VAR)
        if log_file_str:
            log_file = Path(log_file_str)

    # Add file handler if log file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Store metadata on logger for later reference
    logger.step_number = step_number
    logger.total_steps = total_steps
    logger.log_file = log_file

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure console-only logging for subprocess scripts.

    When run as a subprocess, output goes to stdout and the parent
    process captures and logs it. No file handler needed.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance with console-only output
    """
    logger = logging.getLogger("jiflr.pipeline")
    logger.handlers.clear()
    logger.setLevel(level)

    # Console handler only - parent will capture and log to file
    console_handler = TqdmCompatibleHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """
    Get the pipeline logger instance.

    Returns
    -------
    logging.Logger
        The pipeline logger (creates default if not configured)
    """
    logger = logging.getLogger("jiflr.pipeline")

    # If no handlers configured, set up basic console logging
    if not logger.handlers:
        logger = setup_pipeline_logging()

    return logger


# Formatting functions

def header(title: str, step_number: Optional[int] = None, total_steps: Optional[int] = None) -> str:
    """
    Create a standardized header string.

    Parameters
    ----------
    title : str
        Header title text
    step_number : int, optional
        Step number (1-indexed)
    total_steps : int, optional
        Total number of steps

    Returns
    -------
    str
        Formatted header string
    """
    sep = SEPARATOR_CHAR * SEPARATOR_WIDTH

    if step_number is not None and total_steps is not None:
        step_info = f"Step {step_number}/{total_steps}: "
    else:
        step_info = ""

    return f"{sep}\n{step_info}{title}\n{sep}"


def subheader(title: str) -> str:
    """
    Create a standardized subheader string.

    Parameters
    ----------
    title : str
        Subheader title text

    Returns
    -------
    str
        Formatted subheader string
    """
    return f"\n{title}\n{SUBSEPARATOR_CHAR * len(title)}"


def footer(success: bool = True) -> str:
    """
    Create a standardized footer string.

    Parameters
    ----------
    success : bool
        Whether the operation completed successfully

    Returns
    -------
    str
        Formatted footer string
    """
    sep = SUBSEPARATOR_CHAR * SEPARATOR_WIDTH
    status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    return f"{sep}\n{status}\n{sep}"


def indent(text: str, level: int = 1) -> str:
    """
    Indent text by a specified number of levels.

    Parameters
    ----------
    text : str
        Text to indent
    level : int
        Number of indentation levels

    Returns
    -------
    str
        Indented text
    """
    prefix = INDENT * level
    return prefix + text


def item(message: str, level: int = 1) -> str:
    """
    Format a bullet point item.

    Parameters
    ----------
    message : str
        Item text
    level : int
        Indentation level

    Returns
    -------
    str
        Formatted item string
    """
    prefix = INDENT * level
    return f"{prefix}- {message}"


def key_value(key: str, value: str, level: int = 1) -> str:
    """
    Format a key-value pair.

    Parameters
    ----------
    key : str
        Key/label text
    value : str
        Value text
    level : int
        Indentation level

    Returns
    -------
    str
        Formatted key-value string
    """
    prefix = INDENT * level
    return f"{prefix}{key}: {value}"


def pipeline_header(log_file: Optional[Path] = None) -> str:
    """
    Create the main pipeline header with timestamp.

    Parameters
    ----------
    log_file : Path, optional
        Path to log file for display

    Returns
    -------
    str
        Formatted pipeline header
    """
    sep = SEPARATOR_CHAR * SEPARATOR_WIDTH
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    lines = [
        sep,
        "JIFLR Data Processing Pipeline",
        sep,
        key_value("Started", timestamp),
    ]

    if log_file is not None:
        lines.append(key_value("Log file", str(log_file)))

    return "\n".join(lines)


def pipeline_footer(success: bool = True) -> str:
    """
    Create the main pipeline footer.

    Parameters
    ----------
    success : bool
        Whether the pipeline completed successfully

    Returns
    -------
    str
        Formatted pipeline footer
    """
    sep = SEPARATOR_CHAR * SEPARATOR_WIDTH
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if success:
        status = "Pipeline completed successfully!"
    else:
        status = "Pipeline failed."

    return f"\n{sep}\n{status}\n{key_value('Finished', timestamp)}"
