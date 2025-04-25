"""
Heat Network Zone (HNZ) Analysis Utilities

This module provides utility functions for analyzing and comparing
DESNZ heat network zones with Nesta's heat network suitability data.

**Key Features:**
- Adds a `DESNZ_pilot_fraction` column to estimate the fraction of an LSOA covered by heat network zones.
- Computes average suitability scores for LSOAs within and outside heat network zones.
- Calculates Mean Absolute Error (MAE) between DESNZ pilot scores and Nesta suitability scores.

**Functions:**
- `setup_paths`: Determines local or S3 file paths based on input flags.
- `save_gdf_to_gpkg`: Saves a GeoDataFrame as a GeoPackage and uploads it to S3 if needed.
- `_compute_lsoa_coverage_stats`: Computes spatial intersections and LSOA coverage stats.
- `load_transform_hn_geodata`: Load the Heat Networks GeoPackage file, ensure consistent CRS, performs a spatial join with LSOA polygons, and calculates the intersection area.
- `add_DESNZ_pilot_fraction`: Merges heat network zone data with suitability scores and computes the fraction covered.
- `calculate_average_scores_for_thresholds`: Computes average suitability scores for different thresholds of coverage.
- `calculate_mae_for_pilot_score`: Calculates MAE for LSOAs inside and outside pilot heat network zones.
- `calculate_mae_for_all`: Computes overall MAE between DESNZ and Nesta scores.

This module supports comparative spatial analysis of heat network zones and heat network suitability data.
"""

from typing import Tuple, List, Optional
import geopandas as gpd
import polars as pl
import logging
from asf_heat_pump_suitability.utils.save_utils import upload_file_to_s3
from hnz_getters.get_datasets import load_desnz_geodata
from asf_heat_pump_suitability.utils.geo_utils import ensure_crs_match
from config.hnz_config import (
    LSOA_SHP_PATH_LOCAL,
    LSOA_SHP_PATH_S3,
    NESTA_HPS_PARQUET_LOCAL,
    NESTA_HPS_PARQUET_S3,
)
import geopandas as gpd
import os


def setup_paths(read_in_s3: bool) -> dict:
    """
    Set up the paths based on whether to read from S3 or locally.

    Args:
        read_in_s3 (bool): If True, set up paths to read from S3. Otherwise, set up local paths.

    Returns:
        dict: A dictionary containing paths for LSOA_SHP_PATH, and NESTA_HP_SUITABILITY_PARQUET_PATH.
    """
    paths = {}
    if read_in_s3:
        paths["LSOA_SHP_PATH"] = LSOA_SHP_PATH_S3
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = NESTA_HPS_PARQUET_S3

    else:
        paths["LSOA_SHP_PATH"] = LSOA_SHP_PATH_LOCAL
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = NESTA_HPS_PARQUET_LOCAL
    return paths


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
        filename_prefix (str): The prefix used for naming the GPKG file. `filename_prefix` will be joined to `_with_desnz_hn_lsoa.gpkg` to generate full filename.
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
    if save_to_s3:
        upload_file_to_s3(
            local_file_path=gpkg_local_file_path,
            s3_bucket=s3_bucket,
            s3_key_dir=s3_key_dir,
            filename=gpkg_filename,
            subfolder=subfolder,
        )


def _compute_lsoa_coverage_stats(
    desnz_hn_gdf: gpd.GeoDataFrame, lsoa_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Perform a spatial intersection between DESNZ heat network zones and LSOA polygons,
    then calculate the fraction of each LSOA's area that is covered by heat network zones.

    Args:
        desnz_hn_gdf (gpd.GeoDataFrame): DESNZ heat network zones.
        lsoa_gdf (gpd.GeoDataFrame): LSOA polygons.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame containing LSOA codes, intersection geometry, and fraction_covered.
            - List of unique LSOA codes present in the final GeoDataFrame.
    """
    # Remove 'index_right' column if it exists
    if "index_right" in desnz_hn_gdf.columns:
        desnz_hn_gdf = desnz_hn_gdf.drop(columns=["index_right"])

    # Ensure LSOAs have a valid total area column
    lsoa_gdf["total_area"] = lsoa_gdf.geometry.area

    # Perform intersection to determine which LSOAs overlap with heat network zones
    intersection_gdf = gpd.overlay(desnz_hn_gdf, lsoa_gdf, how="intersection")

    # Calculate the fraction of the LSOA that is covered by heat network zones
    intersection_gdf["fraction_covered"] = (
        intersection_gdf["geometry"].area / intersection_gdf["total_area"]
    )
    # Count total rows
    total_rows = len(intersection_gdf)

    # Count how many rows have non-null LSOA21CD
    notna_rows = intersection_gdf["LSOA21CD"].notna().sum()

    # Calculate the number of rows that would be dropped
    dropped_count = total_rows - notna_rows

    # If any rows will be dropped, log a warning
    if dropped_count > 0:
        logging.warning(
            f"{dropped_count} rows in intersection geodataframe has a missing LSOA21CD and will be dropped."
        )
    # Extract unique LSOA codes
    unique_lsoa_codes = intersection_gdf["LSOA21CD"].dropna().unique().tolist()

    return intersection_gdf, unique_lsoa_codes


def load_transform_hn_geodata(
    desnz_hn_gpkg_path: str, lsoa_shp_path: str, layer_name: str
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Load the Heat Networks GeoPackage file, ensure consistent CRS,
    perform a spatial join with LSOA polygons, and calculate the intersection area.

    Args:
        desnz_hn_gpkg_path (str): Path to the GeoPackage file with the Heat Network zones.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        layer_name (str): Layer name in the GeoPackage.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame with LSOA codes added via spatial join.
            - List of unique LSOA codes present in the GeoDataFrame.
    """
    desnz_hn_gdf, lsoa_gdf = load_desnz_geodata(
        gpkg_path=desnz_hn_gpkg_path, shp_path=lsoa_shp_path, layer_name=layer_name
    )
    lsoa_gdf = ensure_crs_match(gdf_1=desnz_hn_gdf, gdf_2=lsoa_gdf)
    return _compute_lsoa_coverage_stats(desnz_hn_gdf, lsoa_gdf)


def add_DESNZ_pilot_fraction(
    la_hp_suitability_scores: pl.DataFrame,
    joined_gdf: gpd.GeoDataFrame,
    optional_threshold: Optional[float] = 0,
) -> Tuple[pl.DataFrame, float, float]:
    """
    Add a 'DESNZ_pilot_fraction' column to the Nesta heat pump suitability data using the fraction of area covered by heat network zones and compute the average Nesta heat network score per LA.

    Args:
        la_hp_suitability_scores (pl.DataFrame): Data containing 'LSOA21CD' and 'HN_N_avg_score_weighted' columns for a single LA.
        joined_gdf (gpd.GeoDataFrame): GeoDataFrame containing 'LSOA21CD' and 'fraction_covered' columns.
        optional_threshold (Optional[float]): Optional threshold for fraction of heat network zone area contained within LA. Defaults to 0.

    Returns:
        Tuple[pl.DataFrame, float, float]:
            - Updated DataFrame with 'DESNZ_pilot_fraction' column added (fraction of local authority area covered by heat network zones).
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_fraction' is non-zero.
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_fraction' is zero.
    """
    # Convert joined_gdf to a Polars DataFrame
    joined_df = pl.DataFrame(
        {
            "LSOA21CD": joined_gdf["LSOA21CD"],
            "fraction_covered": joined_gdf["fraction_covered"],
        }
    )

    # Merge the fraction_covered into la_hp_suitability_scores
    la_hp_suitability_scores = la_hp_suitability_scores.join(
        joined_df, on="LSOA21CD", how="left"
    )

    # Fill NaN values in fraction_covered with 0
    la_hp_suitability_scores = la_hp_suitability_scores.with_columns(
        pl.col("fraction_covered").fill_null(0).alias("DESNZ_pilot_fraction")
    ).drop("fraction_covered")

    # Calculating the average Nesta heat network score for non-zero and zero DESNZ pilot scores
    avg_hn_score_pilot_nonzero = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores,
        hn_zones=True,
        optional_threshold=optional_threshold,
    )
    avg_hn_score_pilot_zero = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores, hn_zones=False
    )
    return la_hp_suitability_scores, avg_hn_score_pilot_nonzero, avg_hn_score_pilot_zero


def calculate_average_scores_for_thresholds(
    la_hp_suitability_scores: pl.DataFrame, thresholds: List[float]
) -> pl.DataFrame:
    """
    Calculate the average 'HN_N_avg_score_weighted' for various thresholds of 'DESNZ_pilot_fraction'.

    Args:
        la_hp_suitability_scores (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'HN_N_avg_score_weighted' columns.
        thresholds (List[float]): List of thresholds to evaluate.

    Returns:
        pl.DataFrame: DataFrame with columns 'DESNZ_pilot_fraction_threshold' and 'HN_N_avg_score_weighted'.
    """
    results = []
    for threshold in thresholds:
        avg_score = _calculate_hn_pilot_average_score(
            la_hp_suitability_scores, hn_zones=True, optional_threshold=threshold
        )
        results.append(
            {
                "DESNZ_pilot_fraction_threshold": threshold,
                "HN_N_avg_score_weighted": avg_score,
            }
        )
    results_df = pl.DataFrame(results)
    return results_df


def _calculate_hn_pilot_average_score(
    la_hp_suitability_scores: pl.DataFrame,
    hn_zones: bool,
    optional_threshold: Optional[float] = 0,
) -> float:
    """
    Calculate the average Nesta Heat Network score for LSOAs in (`DESNZ_pilot_fraction > optional_threshold`) or not in (`DESNZ_pilot_fraction == 0`) DESNZ heat network pilot areas.

    Args:
        la_hp_suitability_scores (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'HN_N_avg_score_weighted' columns for a LA.
        hn_zones (bool): If True, calculate average Nesta heat network score for LSOAs in DESNZ heat network zones. Set to False to calculate the average score for LSOAs not in heat network zones.
        optional_threshold (Optional[float]): The threshold value for 'DESNZ_pilot_fraction'. Defaults to 0. Range: 0-1.

    Returns:
        float: Average 'HN_N_avg_score_weighted' for the filtered rows.
    """
    if hn_zones:
        # Filter for LSOAs in DESNZ heat network zones
        filtered_la_hp_suitability_scores = la_hp_suitability_scores.filter(
            pl.col("DESNZ_pilot_fraction") > optional_threshold
        )
    else:
        # Filter for LSOAs not in DESNZ heat network zones
        filtered_la_hp_suitability_scores = la_hp_suitability_scores.filter(
            pl.col("DESNZ_pilot_fraction") == 0
        )

    # Calculate the average score
    avg_score = filtered_la_hp_suitability_scores["HN_N_avg_score_weighted"].mean()
    return avg_score


def calculate_mae_for_pilot_score(
    hp_suitability_scores_with_desnz: pl.DataFrame,
    hn_zones: bool,
    optional_threshold: Optional[float] = 0,
) -> float:
    """
    Calculate the Mean Absolute Error (MAE) for entries in or not in DESNZ heat network zones.

    Args:
        hp_suitability_scores_with_desnz (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'absolute_error' columns.
        hn_zones (bool): If True, calculate MAE for entries in DESNZ heat network zones. Set to False to calculate MAE for entries not in heat network zones.
        optional_threshold (Optional[float]): The threshold value for 'DESNZ_pilot_fraction'. Defaults to 0. Range: 0-1.

    Returns:
        float: The MAE for the specified condition.
    """
    if hn_zones:
        # Filter for entries in DESNZ heat network zones
        filtered_df = hp_suitability_scores_with_desnz.filter(
            pl.col("DESNZ_pilot_fraction") > optional_threshold
        )
    else:
        # Filter for entries not in DESNZ heat network zones
        filtered_df = hp_suitability_scores_with_desnz.filter(
            pl.col("DESNZ_pilot_fraction") == 0
        )

    # Calculate the MAE
    mae = filtered_df["absolute_error"].mean()
    return mae


def calculate_mae_for_all(
    hp_suitability_scores: pl.DataFrame, desnz_col: str, nesta_hn_score_col: str
) -> Tuple[pl.DataFrame, float]:
    """
    Calculate the Mean Absolute Error (MAE) between the DESNZ pilot score and Nesta's heat network suitability score, add the absolute error column to the DataFrame,
    and log the result.

    Args:
        hp_suitability_scores (pl.DataFrame): DataFrame containing the DESNZ pilot (actual) and Nesta heat network suitability score (predicted) columns.
        desnz_col (str): Name of the column with DESNZ pilot (actual) values.
        nesta_hn_score_col (str): Name of the column with Nesta heat network suitability (predicted) values.

    Returns:
        Tuple[pl.DataFrame, float]:
            - Updated DataFrame with 'absolute_error' column added.
            - The Mean Absolute Error (MAE) between the actual and predicted columns.
    """
    # Calculate absolute error and add it as a new column
    hp_suitability_scores_with_error = hp_suitability_scores.with_columns(
        (pl.col(desnz_col) - pl.col(nesta_hn_score_col)).abs().alias("absolute_error")
    )

    # Calculate the mean of the absolute error
    mae = hp_suitability_scores_with_error["absolute_error"].mean()
    logging.info(
        f"Mean Absolute Error (MAE) for {desnz_col} vs {nesta_hn_score_col}: {mae}"
    )

    return hp_suitability_scores_with_error, mae
