"""
Logging, File Management, and S3 Utilities for evaluation of our Heat Network modelling

This module provides utility functions to:
- Set up logging for scripts.
- Configure local and S3 file paths dynamically.
- Save and optionally upload analysis outputs (e.g., GeoDataFrames) to S3.

**Functions:**
- `setup_logging_and_file_path`: Initialises logging and ensures the log directory exists.
- `setup_paths`: Determines local or S3 file paths based on input flags.
- `optionally_upload_file_to_s3`: Uploads a file to an S3 bucket if required.
- `save_gdf_to_gpkg`: Saves a GeoDataFrame as a GeoPackage and uploads it to S3 if needed.

This module is used in the main script to manage logging, data paths, and output storage efficiently.
"""

import logging
import os
import boto3
import geopandas as gpd
from config.hnz_config import (
    LSOA_SHP_PATH_LOCAL,
    LSOA_SHP_PATH_S3,
    NESTA_HPS_PARQUET_LOCAL,
    NESTA_HPS_PARQUET_S3,
)


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


def setup_paths(read_in_s3: bool) -> dict:
    """
    Set up the paths based on whether to read from S3 or locally.

    Args:
        read_in_s3 (bool): If True, set up paths to read from S3. Otherwise, set up local paths.

    Returns:
        dict: A dictionary containing paths for LIVERPOOL_GPKG_PATH, LSOA_SHP_PATH, and NESTA_HP_SUITABILITY_PARQUET_PATH.
    """
    paths = {}
    if read_in_s3:
        paths["LSOA_SHP_PATH"] = LSOA_SHP_PATH_S3
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = NESTA_HPS_PARQUET_S3

    else:
        paths["LSOA_SHP_PATH"] = LSOA_SHP_PATH_LOCAL
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = NESTA_HPS_PARQUET_LOCAL
    return paths


def optionally_upload_file_to_s3(
    local_file_path: str,
    s3_bucket: str,
    s3_key_dir: str,
    save_to_s3: bool,
    filename: str,
    subfolder: str,
):
    """
    Upload a local file to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): S3 key (path) where the file should be uploaded.
        save_to_s3 (bool): boolean which indicates whether to save or not the
        subfolder (str): Subfolder within S3.
        filename (str): The actual filename to store in S3.
    """
    if save_to_s3:
        s3_client = boto3.client("s3")
        s3_key = f"{s3_key_dir}{subfolder}/{filename}"
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        logging.info(f"File uploaded to s3://{s3_bucket}/{s3_key}")


def save_gdf_to_gpkg(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    filename_prefix: str,
    save_to_s3: bool,
    s3_bucket: str,
    s3_key_dir: str,
    subfolder: str,
):
    """
    Save a GeoDataFrame to a GPKG file, then optionally upload to S3.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to write.
        output_dir (str): Local directory to save the GPKG.
        filename_prefix (str): The prefix used for naming the GPKG file.
        save_to_s3 (bool): Whether to upload the output file to S3.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): The S3 key prefix where files are stored.
        subfolder (str): Optional subfolder within S3 for organization.
    """
    gpkg_filename = (
        f"{filename_prefix.lower().replace(' ', '_')}_with_desnz_hn_lsoa.gpkg"
    )
    gpkg_local_file_path = os.path.join(output_dir, gpkg_filename)

    gdf.to_file(filename=gpkg_local_file_path, driver="GPKG")
    logging.info(f"Saved GPKG for {filename_prefix} to {gpkg_local_file_path}")

    # s3_key = f"{s3_key_dir}{subfolder}/{gpkg_filename}"
    optionally_upload_file_to_s3(
        local_file_path=gpkg_local_file_path,
        s3_bucket=s3_bucket,
        s3_key_dir=s3_key_dir,
        filename=gpkg_filename,
        save_to_s3=save_to_s3,
        subfolder=subfolder,
    )
