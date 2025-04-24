"""
Filename: hnz_getters/get_datasets.py

This module provides functions to load and process datasets for DESNZ heat network zones (HNZ) analysis. It supports local and S3-based data sources and handles formats like Parquet, GeoPackage, Shapefiles, and JSON.

Key Functions:
1. `load_desnz_geodata`: Loads DESNZ heat network polygons and LSOA shapefiles as GeoDataFrames.
2. `load_n_hn_ashp_scores`: Loads Nesta ASHP and HN suitability scores from a Parquet file.
3. `load_la_data`: Loads Local Authority-specific datasets, including:
   - HP suitability scores.
   - LSOA JSON files.
   - Average Nesta HN scores.
"""

import polars as pl
import pandas as pd
import geopandas as gpd
import os
import logging
from asf_heat_pump_suitability.getters import base_getters
import pyogrio
from typing import Tuple, List
import boto3
import json
import io


def load_desnz_geodata(
    gpkg_path: str, shp_path: str, layer_name: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load DESNZ heat network polygons from a GeoPackage and LSOA shapefile.

    Args:
        desnz_hn_gpkg_path (str): Path to the DESNZ Heat Network GeoPackage.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        layer_name (str): Layer name in the GeoPackage.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            - DESNZ heat network zones as a GeoDataFrame.
            - LSOA polygons as a GeoDataFrame.
    """
    hn_gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name)
    lsoa_gdf = gpd.read_file(shp_path)
    return hn_gdf, lsoa_gdf


def load_n_hn_ashp_scores(
    nesta_hps_parquet_path: dict, read_from_s3: bool = False
) -> pd.DataFrame:
    """
    Loads the *full* Parquet containing Nesta ASHP and HN suitability scores (for *all* LAs).
    Returns a Pandas DataFrame with columns like:
      - 'lsoa'
      - 'ASHP_N_avg_score_weighted'
      - 'HN_N_avg_score_weighted'

    Args:
        nesta_hps_parquet_path (dict): Dictionary with "local" and "s3" paths to the Nesta HP suitability Parquet file.
        read_from_s3 (bool): Whether to read the Parquet file from an s3 path or local path.

    Returns:
        pd.DataFrame
    """
    path = (
        nesta_hps_parquet_path["s3"]
        if read_from_s3
        else nesta_hps_parquet_path["local"]
    )
    logging.info(f"Loading global HP suitability data from {path}...")

    df_hps = pd.read_parquet(path)

    needed = {"lsoa", "ASHP_N_avg_score_weighted", "HN_N_avg_score_weighted"}
    if not needed.issubset(df_hps.columns):
        raise ValueError(f"Missing columns {needed} in: {df_hps.columns}")

    logging.info("Successfully loaded HN and ASHP scores.")
    return df_hps


def load_la_data(
    la_name: str,
    input_dir: str,
    s3_bucket: str,
    s3_key_dir: str,
    read_from_s3: bool = False,
) -> Tuple[pd.DataFrame, List[str], pl.DataFrame]:
    """
    Loads LA-specific data files:
      1) A Parquet containing HP suitability scores with DESNZ coverage.
      2) A JSON file listing all LSOAs for the LA.
      3) A Parquet containing average Nesta HN scores across different coverage thresholds.

    Filenames are derived from the LA name, e.g., for 'Birmingham':
      - birmingham_hp_suitability_scores_with_desnz.parquet
      - birmingham_hp_suitability_lsoas.json
      - birmingham_average_scores_by_threshold.parquet

    Args:
        la_name (str): Name of the Local Authority.
        output_dir (str): Path to the directory where the data files are stored.
        s3_bucket (str): Name of the S3 bucket where the data files are stored.
        s3_key_dir (str): Path to the directory within the S3 bucket where the data files are stored.
        read_from_s3 (bool): If True, read files from S3 using boto3. If False, read from local paths.

    Returns:
        Tuple[pd.DataFrame, List[str], pl.DataFrame]:
            - A pandas DataFrame with columns such as 'LSOA21CD', 'DESNZ_pilot_fraction', 'absolute_error'.
            - A list of LSOA codes for the LA.
            - A Polars DataFrame containing average Nesta HN scores vs DESNZ coverage thresholds.

    Raises:
        IOError: If any file fails to load, or if expected columns are missing.
    """
    hp_parquet = f"hp_suitability_scores_with_desnz/{la_name}_hp_suitability_scores_with_desnz.parquet"
    lsoa_json = f"hp_suitability_lsoas/{la_name}_hp_suitability_lsoas.json"
    avg_scores_parquet = f"avg_scores/{la_name}_average_scores_by_threshold.parquet"

    logging.info(f"Loading data files for '{la_name}'...")

    try:
        if read_from_s3:
            # S3 reading approach
            s3_client = boto3.client("s3")

            # 1) HP Parquet
            hp_obj = s3_client.get_object(
                Bucket=s3_bucket, Key=f"{s3_key_dir}{hp_parquet}"
            )
            body = hp_obj["Body"].read()
            buffer = io.BytesIO(body)
            hp_suitability_scores_pd = pd.read_parquet(buffer)

            # 2) LSOA JSON
            lsoa_obj = s3_client.get_object(
                Bucket=s3_bucket, Key=f"{s3_key_dir}{lsoa_json}"
            )
            la_lsoas = json.loads(lsoa_obj["Body"].read())

            # 3) Average scores Parquet
            avg_hn_scores_s3_path = f"s3://{s3_bucket}/{s3_key_dir}{avg_scores_parquet}"
            average_hn_scores_coverage_df = base_getters.get_pl_df_from_parquet(
                avg_hn_scores_s3_path, read_from_s3
            )
        else:
            hp_parquet_local = os.path.join(input_dir, hp_parquet)
            lsoa_json_local = os.path.join(input_dir, lsoa_json)
            avg_scores_local = os.path.join(input_dir, avg_scores_parquet)
            # Local reading approach
            hp_suitability_scores_pd = pd.read_parquet(hp_parquet_local)
            with open(lsoa_json_local, "r") as file:
                la_lsoas = json.load(file)
            average_hn_scores_coverage_df = pl.read_parquet(avg_scores_local)

        # Basic validation
        expected_cols = ["LSOA21CD", "DESNZ_pilot_fraction", "absolute_error"]
        missing_cols = [
            col for col in expected_cols if col not in hp_suitability_scores_pd.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing expected columns for {la_name}: {missing_cols}")

        return hp_suitability_scores_pd, la_lsoas, average_hn_scores_coverage_df

    except Exception as e:
        error_msg = f"Failed to load input data for {la_name} due to: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)
