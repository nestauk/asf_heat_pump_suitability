"""
This script processes and compares DESNZ heat network (HN) and heat pump suitability data for Liverpool, and outputs some files for plotting.

Steps:
1. Load DESNZ HN GeoData and perform a spatial join with LSOA polygons.
2. Process Nesta heat pump suitability scores for Liverpool.
3. Identify LSOAs in DESNZ HN list not in HP suitability data.
4. Calculate and log average scores and Mean Absolute Error (MAE).
5. Save processed data to files.

Outputs:
- 'liverpool_with_desnz_hn_lsoa.gpkg': GeoDataFrame with LSOA codes added to HN zones.
- 'liverpool_hp_suitability_lsoas.json': List of unique LSOA codes in Liverpool.
- 'liverpool_hp_suitability_scores_with_desnz.parquet': Processed suitability scores with DESNZ pilot scores and MAE.
- 'script_output.log': Log file with metrics such as avg HN scores and MAE.
"""

import geopandas as gpd
import pyogrio
import polars as pl
from typing import Tuple, List, Optional
import logging
import json
import os
from asf_heat_pump_suitability import PROJECT_DIR


# Define file paths
LIVERPOOL_GPKG_PATH = "s3://asf-heat-pump-suitability/heat_network_desnz_data/heat-network-zone-map-Liverpool.gpkg"
LSOA_SHP_PATH = "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"
NESTA_HP_SUITABILITY_PARQUET_PATH = "s3://nesta-open-data/asf_heat_pump_suitability/2023Q4/20240925_2023_Q4_EPC_heat_pump_suitability_per_lsoa.parquet"
LA_TO_ANALYSE = "Liverpool"


def setup_logging_and_file_path(
    output_dir: str, log_filename: str = "script_output.log", level: int = logging.INFO
) -> None:
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
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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


def load_hn_geodata(
    desnz_hn_gpkg_path: str, lsoa_shp_path: str
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Load the HN GeoPackage file and perform a spatial join with LSOA polygons to add a column of LSOA codes.

    Args:
        desnz_hn_gpkg_path (str): Path to the GeoPackage file with the HN zones.
        lsoa_shp_path (str): Path to the LSOA shapefile.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame with LSOA codes added via spatial join.
            - List of unique LSOA codes present in the GeoDataFrame.
    """
    # Load DESNZ HN polygons GeoPackage
    desnz_hn_gdf = pyogrio.read_dataframe(
        desnz_hn_gpkg_path, layer="heat-network-zone-map-Liverpool"
    )

    # Load the LSOA polygons file with 'LSOA21CD' codes
    lsoa_gdf = gpd.read_file(lsoa_shp_path)

    # Remove 'index_right' column if it exists
    if "index_right" in desnz_hn_gdf.columns:
        desnz_hn_gdf = desnz_hn_gdf.drop(columns=["index_right"])

    # Ensure both GeoDataFrames use the same CRS
    if desnz_hn_gdf.crs != desnz_hn_gdf.crs:
        lsoa_gdf = lsoa_gdf.to_crs(desnz_hn_gdf.crs)

    # Perform spatial join to match Liverpool polygons with LSOA codes
    joined_gdf = gpd.sjoin(
        desnz_hn_gdf,
        lsoa_gdf[["geometry", "LSOA21CD"]],
        how="left",
        predicate="intersects",
    )

    # Extract unique LSOA codes from the joined GeoDataFrame
    desnz_hn_unique_lsoas = joined_gdf["LSOA21CD"].dropna().unique().tolist()

    return joined_gdf, desnz_hn_unique_lsoas


def process_nesta_hp_suitability_scores(
    nesta_hp_suitability_scores: str, local_authority: str
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Load and process Nesta heat pump suitability data for a specific local authority.

    Args:
        nesta_hp_suitability_scores (str): Path to the Nesta heat pump suitability Parquet file with HN scores.
        local_authority (str): Local authority name to filter the data.

    Returns:
        Tuple[pl.DataFrame, List[str]]:
            - DataFrame containing filtered LSOA codes and 'HN_N_avg_score_weighted' column.
            - List of unique LSOA codes within the local authority.
    """
    # Load the data from the Parquet file
    hp_scores_df = pl.read_parquet(nesta_hp_suitability_scores)

    # Rename 'lsoa' to 'LSOA21CD' if needed to match the GeoDataFrame
    if "lsoa" in hp_scores_df.columns:
        hp_scores_df = hp_scores_df.rename({"lsoa": "LSOA21CD"})

    # Filter by local authority and select relevant columns
    la_filtered_scores = hp_scores_df.filter(
        pl.col("lsoa_name").str.starts_with(local_authority)
    ).select(["LSOA21CD", "HN_N_avg_score_weighted"])

    # Extract unique LSOA codes
    la_unique_lsoas = la_filtered_scores["LSOA21CD"].unique().to_list()

    return la_filtered_scores, la_unique_lsoas


def add_desnz_pilot_score(
    la_hp_suitability_scores: pl.DataFrame, list_of_desnz_hn_lsoas: List[str]
) -> Tuple[pl.DataFrame, float, float]:
    """
    Add a 'DESNZ_pilot_score' column to the Nesta heat pump suitability data and compute average scores.

    Args:
        la_hp_suitability_scores (pl.DataFrame): Data containing 'LSOA21CD' and 'HN_N_avg_score_weighted' columns.
        list_of_desnz_hn_lsoas (List[str]): List of LSOA codes within the DESNZ HN pilot zones.

    Returns:
        Tuple[pl.DataFrame, float, float]:
            - Updated DataFrame with 'DESNZ_pilot_score' column added (1 if in pilot zone, otherwise 0).
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_score' is 1.
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_score' is 0.
    """
    # Add 'DESNZ_pilot_score' column: 1 if LSOA is in DESNZ pilot zone, else 0
    la_hp_suitability_scores = la_hp_suitability_scores.with_columns(
        pl.col("LSOA21CD")
        .is_in(list_of_desnz_hn_lsoas)
        .cast(pl.Int8)
        .alias("DESNZ_pilot_score")
    )
    # Calculate average scores where 'DESNZ_pilot_score' == 1 and == 0
    avg_hn_score_pilot_1 = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores, pilot_score=1
    )
    avg_hn_score_pilot_0 = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores, pilot_score=0
    )
    return la_hp_suitability_scores, avg_hn_score_pilot_1, avg_hn_score_pilot_0


def _calculate_hn_pilot_average_score(
    la_hp_suitability_scores: pl.DataFrame, pilot_score: int
) -> float:
    """
    Calculate the average 'HN_N_avg_score_weighted' for a specified 'DESNZ_pilot_score' value.

    Args:
        la_hp_suitability_scores (pl.DataFrame): DataFrame containing 'DESNZ_pilot_score' and 'HN_N_avg_score_weighted' columns for a LA.
        pilot_score (int): The value of 'DESNZ_pilot_score' to filter by (must be 1 or 0).

    Returns:
        float: The average 'HN_N_avg_score_weighted' for rows with the specified 'DESNZ_pilot_score'.
               Returns 0.0 if there are no rows with the specified score.

    Raises:
        ValueError: If pilot_score is not 1 or 0.
    """
    # Validate that pilot_score is either 1 or 0
    if pilot_score not in {0, 1}:
        raise ValueError("pilot_score must be either 1 or 0")

    # Filter and calculate the average score
    filtered_la_hp_suitability_scores = la_hp_suitability_scores.filter(
        pl.col("DESNZ_pilot_score") == pilot_score
    )
    avg_hn_score_pilot = (
        filtered_la_hp_suitability_scores.select(
            pl.col("HN_N_avg_score_weighted").mean()
        ).item()
        if filtered_la_hp_suitability_scores.height > 0
        else 0.0
    )
    return avg_hn_score_pilot


def calculate_mae_for_pilot_score(
    hp_suitability_scores_with_desnz: pl.DataFrame, score_value: int
) -> float:
    """
    Calculate and log the Mean Absolute Error (MAE) for entries with a specified 'DESNZ_pilot_score'.

    Args:
        hp_suitability_scores_with_desnz (pl.DataFrame): DataFrame containing 'DESNZ_pilot_score' and 'absolute_error' columns.
        score_value (int): The value of 'DESNZ_pilot_score' (must be 1 or 0) for which to calculate the MAE.

    Returns:
        float: The MAE for the specified 'DESNZ_pilot_score'.

    Raises:
        ValueError: If score_value is not 1 or 0.
    """
    # Validate that score_value is either 1 or 0
    if score_value not in {0, 1}:
        raise ValueError("score_value must be either 1 or 0")

    mae = hp_suitability_scores_with_desnz.filter(
        pl.col("DESNZ_pilot_score") == score_value
    )["absolute_error"].mean()

    logging.info(
        f"Mean Absolute Error (MAE) for DESNZ_pilot_score = {score_value}: {mae}"
    )
    return mae


def calculate_mae_for_all(
    hp_suitability_scores: pl.DataFrame, desnz_col: str, nesta_hn_score_col: str
) -> Tuple[pl.DataFrame, float]:
    """
    Calculate the Mean Absolute Error (MAE) between the DESNZ pilot score and Nesta's HN suitability score, add the absolute error column to the DataFrame,
    and log the result.

    Args:
        hp_suitability_scores_with_desnz (pl.DataFrame): DataFrame containing the DESNZ pilot (actual) and Nesta HN suitability score (predicted) columns.
        desnz_col (str): Name of the column with DESNZ pilot (actual) values.
        nesta_hn_score_col (str): Name of the column with Nesta HN suitability (predicted) values.

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


def main():
    # Define the output directory and set up logging
    output_dir = os.path.join(
        PROJECT_DIR, "asf_heat_pump_suitability/analysis/hn_zones/output_data/"
    )
    setup_logging_and_file_path(output_dir)

    # Load the data and perform spatial join
    liverpool_with_desnz_hn_lsoa, list_of_liverpool_desnz_hn_lsoas = load_hn_geodata(
        LIVERPOOL_GPKG_PATH, LSOA_SHP_PATH
    )
    logging.info("Loaded Liverpool with DESNZ HN LSOA data.")
    liverpool_with_desnz_hn_lsoa.to_file(
        os.path.join(output_dir, "liverpool_with_desnz_hn_lsoa.gpkg"), driver="GPKG"
    )

    # Process Nesta heat pump suitability scores
    liverpool_hp_suitability_scores, liverpool_hp_suitability_lsoas = (
        process_nesta_hp_suitability_scores(
            NESTA_HP_SUITABILITY_PARQUET_PATH, LA_TO_ANALYSE
        )
    )
    logging.info("Processed Nesta HP suitability scores for Liverpool.")

    with open(
        os.path.join(output_dir, "liverpool_hp_suitability_lsoas.json"), "w"
    ) as file:
        json.dump(liverpool_hp_suitability_lsoas, file)

    # Check LSOAs not in HP suitability scores
    not_in_hp_suitability = set(list_of_liverpool_desnz_hn_lsoas) - set(
        liverpool_hp_suitability_lsoas
    )
    logging.info(
        f"number of LSOAs in DESNZ list not in HP suitability data: {len(not_in_hp_suitability)}"
    )

    # Calculate and log average score
    avg_hn_score = liverpool_hp_suitability_scores["HN_N_avg_score_weighted"].mean()
    logging.info(f"Average HN_N_avg_score_weighted: {avg_hn_score}")

    # Add DESNZ pilot score and calculate averages
    (
        liverpool_hp_suitability_scores_with_desnz,
        avg_hn_score_pilot_1,
        avg_hn_score_pilot_0,
    ) = add_desnz_pilot_score(
        liverpool_hp_suitability_scores, list_of_liverpool_desnz_hn_lsoas
    )
    logging.info(
        f"Avg HN_N_avg_score_weighted for DESNZ_pilot_score = 1: {avg_hn_score_pilot_1}"
    )
    logging.info(
        f"Avg HN_N_avg_score_weighted for DESNZ_pilot_score = 0: {avg_hn_score_pilot_0}"
    )

    # Calculate and log Mean Absolute Error (MAE)
    liverpool_hp_suitability_scores_with_desnz, mae_all = calculate_mae_for_all(
        liverpool_hp_suitability_scores_with_desnz,
        "DESNZ_pilot_score",
        "HN_N_avg_score_weighted",
    )
    mae_pilot_1 = calculate_mae_for_pilot_score(
        liverpool_hp_suitability_scores_with_desnz, score_value=1
    )
    mae_pilot_0 = calculate_mae_for_pilot_score(
        liverpool_hp_suitability_scores_with_desnz, score_value=0
    )

    liverpool_hp_suitability_scores_with_desnz.write_parquet(
        os.path.join(output_dir, "liverpool_hp_suitability_scores_with_desnz.parquet")
    )


if __name__ == "__main__":
    main()
