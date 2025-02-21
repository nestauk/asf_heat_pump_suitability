"""
Logging of our Heat Network modelling

This module provides a utility function to:
- Set up logging for scripts.

**Function:**
- `setup_logging_and_file_path`: Initialises logging and ensures the log directory exists.

This module is used in the main script to manage logging.
"""

import logging
import os


def setup_logging_and_file_path(
    output_dir: str, log_filename: str = "script_output.log", level: int = logging.INFO
):
    """
    Set up logging configuration to log messages to both a file and the console.

    Args:
        output_dir (str): Path to the directory where the log file should be saved.
        log_filename (str): Name of the log file. Defaults to "script_output.log".
        level (int): Logging level, e.g., logging.INFO or logging.DEBUG. Defaults to logging.INFO.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path to the log file
    log_file_path = os.path.join(output_dir, log_filename)

    # Clear existing logging handlers if any
    logging.getLogger().handlers.clear()

    # Set up logging configuration
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),  # Overwrite log file each run
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging setup complete. Logs are saved to {log_file_path}.")
