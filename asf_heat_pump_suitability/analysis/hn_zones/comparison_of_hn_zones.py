"""
This script analyses DESNZ heat network zones and Nesta's heat pump suitability data for Liverpool, performing spatial joins, statistical analysis, and exporting processed results.

**Key Steps:**
1. **Spatial Analysis**:
   - Loads DESNZ heat network zones and LSOA polygons, ensuring consistent CRS.
   - Performs spatial joins to calculate the fraction of each LSOA covered by heat network zones.

2. **Nesta Data Processing**:
   - Filters Nesta heat pump suitability scores for Liverpool's LSOAs.
   - Identifies LSOAs which are outside selected LA.

3. **Statistical Metrics**:
   - Calculates average Nesta suitability scores for LSOAs covered and not covered by heat network zones.
   - Computes Mean Absolute Error (MAE) between DESNZ and Nesta scores.

4. **Data Export**:
   - Outputs GeoDataFrame with coverage data, LSOA lists, and processed suitability scores to GeoPackage, JSON, Parquet, and CSV formats.
   - Logs detailed metrics and analysis steps.

**Outputs**:
- GeoPackage: 'liverpool_with_desnz_hn_lsoa.gpkg'
- JSON: 'liverpool_hp_suitability_lsoas.json'
- Parquet: 'liverpool_hp_suitability_scores_with_desnz'
- Log: 'script_output.log'

**How to Run the Script**:
To run the script, use the following command:
python comparison_of_hn_zones.py [--optional_threshold OPTIONAL_THRESHOLD] [--read_in_s3 READ_IN_S3] [--save_to_s3 SAVE_TO_S3]

Example:
    # Run the script with default settings (local files, no threshold)
    python comparison_of_hn_zones.py

    # Run the script with a specific threshold
    python comparison_of_hn_zones.py --optional_threshold 0.1

    # Run the script with files from S3
    python comparison_of_hn_zones.py --read_in_s3 True

    # Run the script and save outputs to S3
    python comparison_of_hn_zones.py --save_to_s3 True

    # Run the script with a specific threshold, read from S3, and save to S3
    python comparison_of_hn_zones.py --optional_threshold 0.1 --read_in_s3 True --save_to_s3 True
"""

import geopandas as gpd
import pyogrio
import polars as pl
from typing import Tuple, List, Optional
import logging
import json
import os
from asf_heat_pump_suitability import PROJECT_DIR
import argparse
import boto3


# Define file paths
LA_TO_ANALYSE = "Liverpool"
S3_BUCKET = "asf-heat-pump-suitability"
S3_KEY_DIR = "evaluation/desnz_hn_zone_scores/"


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
        paths["LIVERPOOL_GPKG_PATH"] = (
            "s3://asf-heat-pump-suitability/heat_network_desnz_data/heat-network-zone-map-Liverpool.gpkg"
        )
        paths["LSOA_SHP_PATH"] = (
            "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"
        )
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = (
            "s3://nesta-open-data/asf_heat_pump_suitability/2023Q4/20240925_2023_Q4_EPC_heat_pump_suitability_per_lsoa.parquet"
        )

    else:
        paths["LIVERPOOL_GPKG_PATH"] = os.path.join(
            PROJECT_DIR,
            "asf_heat_pump_suitability/analysis/hn_zones/input_data/desnz_heat_network_zone_maps/heat-network-zone-map-Liverpool.gpkg",
        )
        paths["LSOA_SHP_PATH"] = os.path.join(
            PROJECT_DIR,
            "asf_heat_pump_suitability/analysis/hn_zones/input_data/lsoa_shape_file/LSOA_2021_EW_BFE_V9.shp",
        )
        paths["NESTA_HP_SUITABILITY_PARQUET_PATH"] = os.path.join(
            PROJECT_DIR,
            "asf_heat_pump_suitability/analysis/hn_zones/input_data/nesta_heat_network_suitability/20240925_2023_Q4_EPC_heat_pump_suitability_per_lsoa.parquet",
        )
    return paths


def optionally_upload_file_to_s3(
    local_file_path: str, s3_bucket: str, s3_key: str, save_to_s3: bool
) -> None:
    """
    Upload a local file to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        s3_bucket (str): Name of the S3 bucket.
        s3_key (str): S3 key (path) where the file should be uploaded.
    """
    if save_to_s3:
        s3_client = boto3.client("s3")
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        logging.info(f"File uploaded to s3://{s3_bucket}/{s3_key}")


def load_transform_hn_geodata(
    desnz_hn_gpkg_path: str, lsoa_shp_path: str
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Load the Heat Networks GeoPackage file, perform a spatial join with LSOA polygons to add a column of LSOA codes and calculates the intersection area and the fraction of LSOA area covered by heat network zones.

    Args:
        desnz_hn_gpkg_path (str): Path to the GeoPackage file with the Heat Network zones.
        lsoa_shp_path (str): Path to the LSOA shapefile.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame with LSOA codes added via spatial join.
            - List of unique LSOA codes present in the GeoDataFrame.
    """
    # Load DESNZ heat network polygons GeoPackage
    desnz_hn_gdf = pyogrio.read_dataframe(
        desnz_hn_gpkg_path, layer="heat-network-zone-map-Liverpool"
    )

    # Load the LSOA polygons file with 'LSOA21CD' codes
    lsoa_gdf = gpd.read_file(lsoa_shp_path)

    # Remove 'index_right' column if it exists
    if "index_right" in desnz_hn_gdf.columns:
        desnz_hn_gdf = desnz_hn_gdf.drop(columns=["index_right"])

    # Ensure both GeoDataFrames use the same CRS
    if desnz_hn_gdf.crs != lsoa_gdf.crs:
        lsoa_gdf = lsoa_gdf.to_crs(desnz_hn_gdf.crs)
    # Calculate the total area of the LSOA
    lsoa_gdf["total_area"] = lsoa_gdf.geometry.area
    # Get all intersections between DESNZ heat network zones and LSOAs
    joined_gdf = gpd.overlay(desnz_hn_gdf, lsoa_gdf, how="intersection")
    # Calculate area of intersections
    joined_gdf["fraction_covered"] = (
        joined_gdf["geometry"].area / joined_gdf["total_area"]
    )
    desnz_hn_unique_lsoas = joined_gdf["LSOA21CD"].dropna().unique().tolist()
    return joined_gdf, desnz_hn_unique_lsoas


def filter_la_nesta_hp_scores(
    nesta_hp_suitability_scores: str, local_authority: str
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Load and process Nesta heat pump suitability data for a specific local authority.

    Args:
        nesta_hp_suitability_scores (str): Path to the Nesta heat pump suitability Parquet file with heat network scores.
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

    # Extract unique LSOA codes for LA
    la_unique_lsoas = la_filtered_scores["LSOA21CD"].unique().to_list()

    return la_filtered_scores, la_unique_lsoas


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


if __name__ == "__main__":
    # Argument parser for optional threshold
    parser = argparse.ArgumentParser(
        description="Process heat network zones and calculate scores."
    )
    parser.add_argument(
        "--optional_threshold",
        type=float,
        default=0.0,
        help="Optional threshold for DESNZ pilot fraction for calculating average Nesta heat network score. Range: 0-1.",
    )
    parser.add_argument(
        "--read_in_s3",
        type=bool,
        default=False,
        help="Read in the input files from S3.",
    )
    parser.add_argument(
        "--save_to_s3",
        type=bool,
        default=False,
        help="Save the output files to S3.",
    )
    args = parser.parse_args()
    # Extract the optional_threshold value from args
    optional_threshold = args.optional_threshold
    save_to_s3 = args.save_to_s3
    read_in_s3 = args.read_in_s3

    # Set up paths based on the read_from_s3 flag
    paths = setup_paths(read_in_s3)

    # Access the paths from the dictionary
    LIVERPOOL_GPKG_PATH = paths["LIVERPOOL_GPKG_PATH"]
    LSOA_SHP_PATH = paths["LSOA_SHP_PATH"]
    NESTA_HP_SUITABILITY_PARQUET_PATH = paths["NESTA_HP_SUITABILITY_PARQUET_PATH"]

    # Define the output directory and set up logging
    output_dir = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_data/")

    setup_logging_and_file_path(output_dir=output_dir)

    # Load the data and perform spatial join
    liverpool_with_desnz_hn_lsoa, list_of_liverpool_desnz_hn_lsoas = (
        load_transform_hn_geodata(
            desnz_hn_gpkg_path=LIVERPOOL_GPKG_PATH, lsoa_shp_path=LSOA_SHP_PATH
        )
    )

    logging.info("Loaded Liverpool with DESNZ heat network LSOA data.")

    liv_desnz_hn_filename = "liverpool_with_desnz_hn_lsoa.gpkg"
    liv_desnz_hn_local_file_path = os.path.join(output_dir, liv_desnz_hn_filename)
    liverpool_with_desnz_hn_lsoa.to_file(
        filename=liv_desnz_hn_local_file_path,
        driver="GPKG",
    )
    liv_desnz_hn_s3_key = f"{S3_KEY_DIR}{liv_desnz_hn_filename}"
    optionally_upload_file_to_s3(
        local_file_path=liv_desnz_hn_local_file_path,
        s3_bucket=S3_BUCKET,
        s3_key=liv_desnz_hn_s3_key,
        save_to_s3=save_to_s3,
    )

    # Process Nesta heat pump suitability scores
    liverpool_hp_suitability_scores, liverpool_hp_suitability_lsoas = (
        filter_la_nesta_hp_scores(
            nesta_hp_suitability_scores=NESTA_HP_SUITABILITY_PARQUET_PATH,
            local_authority=LA_TO_ANALYSE,
        )
    )
    logging.info("Processed Nesta HP suitability scores for Liverpool.")
    # Define JSON file name and paths
    lsoas_json_filename = "liverpool_hp_suitability_lsoas.json"
    lsoas_json_local_file_path = os.path.join(output_dir, lsoas_json_filename)
    lsoas_json_s3_key = f"{S3_KEY_DIR}{lsoas_json_filename}"

    with open(os.path.join(output_dir, lsoas_json_filename), "w") as file:
        json.dump(liverpool_hp_suitability_lsoas, file)

    optionally_upload_file_to_s3(
        local_file_path=lsoas_json_local_file_path,
        s3_bucket=S3_BUCKET,
        s3_key=lsoas_json_s3_key,
        save_to_s3=save_to_s3,
    )

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
    # Define the list of thresholds
    thresholds = [
        round(i * 0.05, 2) for i in range(0, 20)
    ]  # Generates [0.0, 0.05, ..., 0.95]

    # Add DESNZ pilot score and calculate averages
    (
        liverpool_hp_suitability_scores_with_desnz,
        avg_hn_score_pilot_nonzero,
        avg_hn_score_pilot_zero,
    ) = add_DESNZ_pilot_fraction(
        la_hp_suitability_scores=liverpool_hp_suitability_scores,
        joined_gdf=liverpool_with_desnz_hn_lsoa,
        optional_threshold=optional_threshold,
    )
    logging.info(
        f"Avg HN_N_avg_score_weighted for DESNZ_pilot_fraction when > {optional_threshold}: {avg_hn_score_pilot_nonzero}"
    )
    logging.info(
        f"Avg HN_N_avg_score_weighted for DESNZ_pilot_fraction = 0: {avg_hn_score_pilot_zero}"
    )

    # Calculate average scores for different thresholds
    average_scores_df = calculate_average_scores_for_thresholds(
        la_hp_suitability_scores=liverpool_hp_suitability_scores_with_desnz,
        thresholds=thresholds,
    )

    # Save the results to Parquet
    avg_score_parquet_filename = "average_scores_by_threshold.parquet"
    avg_score_parquet_filepath = os.path.join(output_dir, avg_score_parquet_filename)
    average_scores_df.write_parquet(avg_score_parquet_filepath)
    optionally_upload_file_to_s3(
        local_file_path=avg_score_parquet_filepath,
        s3_bucket=S3_BUCKET,
        s3_key=f"{S3_KEY_DIR}{avg_score_parquet_filename}",
        save_to_s3=save_to_s3,
    )

    # Calculate and log Mean Absolute Error (MAE)
    liverpool_hp_suitability_scores_with_desnz, mae_all = calculate_mae_for_all(
        hp_suitability_scores=liverpool_hp_suitability_scores_with_desnz,
        desnz_col="DESNZ_pilot_fraction",
        nesta_hn_score_col="HN_N_avg_score_weighted",
    )
    mae_pilot_non_zero = calculate_mae_for_pilot_score(
        hp_suitability_scores_with_desnz=liverpool_hp_suitability_scores_with_desnz,
        hn_zones=True,
    )
    logging.info(
        f"Mean Absolute Error (MAE) for DESNZ_pilot_fraction > 0: {mae_pilot_non_zero}"
    )
    mae_pilot_zero = calculate_mae_for_pilot_score(
        hp_suitability_scores_with_desnz=liverpool_hp_suitability_scores_with_desnz,
        hn_zones=False,
    )
    logging.info(
        f"Mean Absolute Error (MAE) for DESNZ_pilot_fraction = 0: {mae_pilot_zero}"
    )
    # Define Parquet file name and paths for MAE results
    mae_parquet_filename = "liverpool_hp_suitability_scores_with_desnz.parquet"
    mae_parquet_local_file_path = os.path.join(output_dir, mae_parquet_filename)
    mae_parquet_s3_key = f"{S3_KEY_DIR}{mae_parquet_filename}"
    liverpool_hp_suitability_scores_with_desnz.write_parquet(
        mae_parquet_local_file_path
    )

    optionally_upload_file_to_s3(
        local_file_path=mae_parquet_local_file_path,
        s3_bucket=S3_BUCKET,
        s3_key=mae_parquet_s3_key,
        save_to_s3=save_to_s3,
    )
    # Write to CSV file
    liverpool_hp_suitability_scores_with_desnz.write_csv(
        os.path.join(output_dir, "liverpool_hp_suitability_scores_with_desnz.csv")
    )
    logging.info("Saved processed data to this output directory:")
    logging.info(output_dir)
