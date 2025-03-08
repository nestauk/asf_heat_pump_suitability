import os
import json
import logging
import boto3
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
from typing import Tuple, List

from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability.getters.base_getters import get_df_from_parquet
from config.hnz_config import (
    OUTPUT_DIR,  # "outputs/hn_zones/output_data/"
    S3_BUCKET,
    S3_KEY_DIR,
)

####################
# 1. Configurations
####################
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
INPUT_DIR = OUTPUT_DIR  # your final data files are here


# This is the path to the *national* LSOA shapefile (either local or S3).
# If your script can read from S3, handle that similarly to your existing code.
LSOA_SHP_PATH = (
    "s3://asf-heat-pump-suitability/source_data/"
    "Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/"
    "LSOA_2021_EW_BFE_V9.shp"
)

# Target CRS (ensure LSOA shapefiles and GPKGs are converted if they differ)
TARGET_CRS = "EPSG:27700"


def load_la_data(
    la_name: str, read_from_s3: bool = False
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
        read_from_s3 (bool): If True, read files from S3 using boto3. If False, read from local paths.

    Returns:
        Tuple[pd.DataFrame, List[str], pl.DataFrame]:
            - A pandas DataFrame with columns such as 'LSOA21CD', 'DESNZ_pilot_fraction', 'absolute_error'.
            - A list of LSOA codes for the LA.
            - A Polars DataFrame containing average Nesta HN scores vs DESNZ coverage thresholds.

    Raises:
        IOError: If any file fails to load, or if expected columns are missing.
    """
    la_snake = la_name.lower().replace(" ", "_")
    hp_parquet = f"{la_snake}_hp_suitability_scores_with_desnz.parquet"
    lsoa_json = f"{la_snake}_hp_suitability_lsoas.json"
    avg_scores_parquet = f"{la_snake}_average_scores_by_threshold.parquet"

    hp_parquet_local = os.path.join(INPUT_DIR, hp_parquet)
    lsoa_json_local = os.path.join(INPUT_DIR, lsoa_json)
    avg_scores_local = os.path.join(INPUT_DIR, avg_scores_parquet)

    logging.info(f"Loading data files for '{la_name}'...")

    try:
        if read_from_s3:
            # S3 reading approach
            s3_client = boto3.client("s3")

            # 1) HP Parquet
            hp_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=f"{S3_KEY_DIR}{hp_parquet}"
            )
            hp_suitability_scores_pd = pd.read_parquet(hp_obj["Body"])

            # 2) LSOA JSON
            lsoa_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=f"{S3_KEY_DIR}{lsoa_json}"
            )
            la_lsoas = json.loads(lsoa_obj["Body"].read())

            # 3) Average scores Parquet
            avg_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=f"{S3_KEY_DIR}{avg_scores_parquet}"
            )
            average_hn_scores_coverage_df = pl.read_parquet(avg_obj["Body"])
        else:
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


def preprocess_data(
    la_lsoas: List[str], lsoa_shp_path: str = LSOA_SHP_PATH
) -> gpd.GeoDataFrame:
    """
    Loads and preprocesses LSOA geospatial data by:
      - Reading a shapefile (can be large).
      - Converting to a target CRS (e.g. EPSG:27700) if necessary.
      - Filtering LSOAs to just those that belong to the specified local authority.

    Args:
        la_lsoas (List[str]): List of LSOA codes for the current local authority.
        lsoa_shp_path (str): Path (local or S3) to the national LSOA shapefile.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filtered LSOA geometries in the target CRS.

    Raises:
        ValueError: If no matching LSOAs are found in the shapefile.
    """
    logging.info("Preprocessing LSOA geometries...")
    lsoa_gdf = gpd.read_file(lsoa_shp_path)

    # Reproject if needed
    if lsoa_gdf.crs != TARGET_CRS:
        logging.info(f"Converting shapefile to {TARGET_CRS}")
        lsoa_gdf = lsoa_gdf.to_crs(TARGET_CRS)

    la_lsoa_geometries_gdf = lsoa_gdf[lsoa_gdf["LSOA21CD"].isin(la_lsoas)]
    if la_lsoa_geometries_gdf.empty:
        raise ValueError(
            "No LSOAs found for this local authority in the provided LSOA shapefile."
        )

    return la_lsoa_geometries_gdf


def merge_data(
    hp_suitability_scores_pd: pd.DataFrame,
    la_lsoa_geometries_gdf: gpd.GeoDataFrame,
    la_name: str,
) -> gpd.GeoDataFrame:
    """
    Merges the LA's heat pump suitability scores (pandas DataFrame) with its LSOA geometries (GeoDataFrame).
    Ensures the final merged result is also a GeoDataFrame in the target CRS.

    Args:
        hp_suitability_scores_pd (pd.DataFrame): DataFrame containing columns like 'LSOA21CD', 'DESNZ_pilot_fraction'.
        la_lsoa_geometries_gdf (gpd.GeoDataFrame): LSOA geometry data for the LA.
        la_name (str): Local Authority name, used for logging.

    Returns:
        gpd.GeoDataFrame: Merged data with geometry.

    Raises:
        ValueError: If the merge results in empty data or missing geometry.
    """
    logging.info(f"Merging suitability data with geometries for {la_name}...")
    merged = hp_suitability_scores_pd.merge(
        la_lsoa_geometries_gdf, on="LSOA21CD", how="left"
    )

    if merged.empty:
        raise ValueError(
            f"Merging resulted in an empty DataFrame for {la_name}. Check LSOA codes."
        )
    if merged["geometry"].isna().any():
        raise ValueError(
            f"Some records are missing geometry after merging for {la_name}."
        )

    la_hp_suitability_gdf = gpd.GeoDataFrame(
        merged, geometry="geometry", crs=TARGET_CRS
    )
    return la_hp_suitability_gdf


plymouth_hp_suitability_scores, plymouth_lsoas, plymouth_avg_hn_scores_coverage_df = (
    load_la_data("Plymouth")
)
plymouth_geometry_gdf = preprocess_data(plymouth_lsoas)

print(plymouth_hp_suitability_scores.head())

print(plymouth_geometry_gdf.head())

plymouth_hp_suitability_gdf = merge_data(
    plymouth_hp_suitability_scores, plymouth_geometry_gdf, "Plymouth"
)


# Convert DESNZ_pilot_fraction to a percentage and rename the column to "percentage"
plymouth_hp_suitability_gdf["% of area covered by a DESNZ HN zone"] = np.round(
    plymouth_hp_suitability_gdf["DESNZ_pilot_fraction"] * 100, 1
)

# Drop the old column if you no longer need it
plymouth_hp_suitability_gdf = plymouth_hp_suitability_gdf.drop(
    columns=["DESNZ_pilot_fraction"]
)

# New column for absolute_error when percentage == 0, else NaN
plymouth_hp_suitability_gdf["Absolute error when a DESNZ HN zone is absent"] = np.where(
    plymouth_hp_suitability_gdf["% of area covered by a DESNZ HN zone"] == 0,
    np.round(
        plymouth_hp_suitability_gdf["absolute_error"], 2
    ),  # Round to 2 decimal places
    "N/A",
)

# New column for absolute_error when percentage > 0, else NaN
plymouth_hp_suitability_gdf["Absolute error when a DESNZ HN zone is present"] = (
    np.where(
        plymouth_hp_suitability_gdf["% of area covered by a DESNZ HN zone"] > 0,
        np.round(
            plymouth_hp_suitability_gdf["absolute_error"], 2
        ),  # Round to 2 decimal places
        "N/A",
    )
)

print(plymouth_hp_suitability_gdf.head())

# Reproject to WGS84 (EPSG:4326)
plymouth_hp_suitability_gdf = plymouth_hp_suitability_gdf.to_crs(epsg=4326)

# Define the output directory
OUTPUT_DATA_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/flourish_prepped_data/")

# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# Save the reshaped dataframe to a CSV file
output_file_path = os.path.join(
    OUTPUT_DATA_DIR, "plymouth_flourish_output_070325.geojson"
)

plymouth_hp_suitability_gdf.to_file(output_file_path, driver="GeoJSON")
