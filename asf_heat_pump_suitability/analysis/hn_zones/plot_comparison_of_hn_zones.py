"""
This script processes and visualises DESNZ heat network (HN) and Nesta heat pump suitability data for Liverpool.

Steps:
1. Load and preprocess data:
   - Read Parquet, CSV, and GeoPackage files.
   - Extract unique LSOA geometries.
   - Load LSOA geometries and convert CRS if needed.
   - Filter Liverpool LSOAs.

2. Plot Liverpool LSOA geometries and save as PNG.

3. Merge suitability scores with geometries and convert to GeoDataFrame.

4. Plot and save overlay of DESNZ Pilot Score > 0 on Liverpool heat network zones.

5. Define and use a function to plot absolute error maps by DESNZ Pilot Score, saving the plots as PNG files.

6. Load and plot average Nesta HN scores against DESNZ Pilot Fraction threshold along with plotting the best fit line and calculating the R^{2}.

Usage:
    python plot_comparison_of_hn_zones.py [--read_from_s3]

Arguments:
    --read_from_s3 (bool): If True, read the input files from S3 bucket. Defaults to False.

Example:
    # Run the script with local files
    python plot_comparison_of_hn_zones.py

    # Run the script with files from S3
    python plot_comparison_of_hn_zones.py --read_from_s3 True
"""

import geopandas as gpd
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import pyogrio
import json
import os
import logging
from asf_heat_pump_suitability import PROJECT_DIR
import numpy as np
from sklearn.metrics import r2_score
import argparse
import boto3

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Define local directories and file paths
INPUT_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_data/")
LOCAL_LIVERPOOL_HP_SUITABILITY_PARQUET = os.path.join(
    INPUT_DIR, "liverpool_hp_suitability_scores_with_desnz.parquet"
)
LOCAL_AVERAGE_HN_SCORES_COVERAGE_PARQUET = os.path.join(
    INPUT_DIR, "average_scores_by_threshold.parquet"
)
LOCAL_LSOA_JSON_PATH = os.path.join(INPUT_DIR, "liverpool_hp_suitability_lsoas.json")

LSOA_SHP_PATH = "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"
LIVERPOOL_GPKG_PATH = "s3://asf-heat-pump-suitability/heat_network_desnz_data/heat-network-zone-map-Liverpool.gpkg"

# Define S3 variables and paths
S3_BUCKET = "asf-heat-pump-suitability"
S3_KEY_DIR = "evaluation/desnz_hn_zone_scores/"
S3_KEY_LIVERPOOL_HP_SUITABILITY_PARQUET = (
    f"{S3_KEY_DIR}liverpool_hp_suitability_scores_with_desnz.parquet"
)
S3_KEY_AVERAGE_HN_SCORES_COVERAGE_PARQUET = (
    f"{S3_KEY_DIR}average_scores_by_threshold.parquet"
)
S3_KEY_LSOA_JSON_PATH = f"{S3_KEY_DIR}liverpool_hp_suitability_lsoas.json"
# Define output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_plots/")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define a variable for target CRS to avoid hardcoding
TARGET_CRS = "EPSG:27700"


def load_data(read_from_s3: bool = False) -> tuple:
    """
    Load the necessary data files and return them.

    Args:
        read_from_s3 (bool): If True, read the input files from S3. If False, read the input files locally.

    Returns:
        Tuple:
            - pl.DataFrame: DataFrame with Liverpool heat pump suitability scores.
            - List[str]: List of LSOA codes for Liverpool.
            - pl.DataFrame: DataFrame with the average heat network scores for different area thresholds (threshold is the fraction of an LSOA which is covered by a DESNZ HN zone).
    Raises:
        IOError: If any of the data files cannot be loaded.
    """
    try:
        logging.info("Loading data files...")
        if read_from_s3:
            # Read from S3
            s3_client = boto3.client("s3")

            # Read Liverpool heat pump suitability scores
            liv_hps_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=S3_KEY_LIVERPOOL_HP_SUITABILITY_PARQUET
            )
            liverpool_hp_suitability_scores_pd = pd.read_parquet(liv_hps_obj["Body"])

            # Read LSOA JSON
            liv_hps_lsoas_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=S3_KEY_LSOA_JSON_PATH
            )
            liverpool_lsoas = json.loads(liv_hps_lsoas_obj["Body"].read())

            # Read average HN scores coverage
            avg_hn_scores_obj = s3_client.get_object(
                Bucket=S3_BUCKET, Key=S3_KEY_AVERAGE_HN_SCORES_COVERAGE_PARQUET
            )
            average_hn_scores_coverage_df = pl.read_parquet(avg_hn_scores_obj["Body"])
        else:
            liverpool_hp_suitability_scores_pd = pd.read_parquet(
                LOCAL_LIVERPOOL_HP_SUITABILITY_PARQUET
            )
            with open(LOCAL_LSOA_JSON_PATH, "r") as file:
                liverpool_lsoas = json.load(file)
            average_hn_scores_coverage_df = pl.read_parquet(
                LOCAL_AVERAGE_HN_SCORES_COVERAGE_PARQUET
            )

        # Basic validation: Check expected columns in the main DataFrame
        expected_cols = ["LSOA21CD", "DESNZ_pilot_fraction", "absolute_error"]
        missing_cols = [
            col
            for col in expected_cols
            if col not in liverpool_hp_suitability_scores_pd.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing expected columns in hp suitability df: {missing_cols}"
            )

        return (
            liverpool_hp_suitability_scores_pd,
            liverpool_lsoas,
            average_hn_scores_coverage_df,
        )
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise IOError(f"Failed to load input data due to: {e}")


def preprocess_data(
    liverpool_lsoas: list,
    lsoa_shp_path: str = LSOA_SHP_PATH,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Load and preprocess LSOA geospatial data by extracting unique LSOA geometries and filtering Liverpool LSOAs.

    Args:
        liverpool_lsoas (list): List of LSOA codes for Liverpool.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        target_crs (str): The target CRS to ensure consistent geospatial analysis.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filtered Liverpool LSOA geometries in the given CRS.
    """
    logging.info("Preprocessing LSOA geometries...")
    lsoa_geometries_gdf = gpd.read_file(lsoa_shp_path)
    if lsoa_geometries_gdf.crs != target_crs:
        logging.info(f"Converting to {target_crs}")
        lsoa_geometries_gdf = lsoa_geometries_gdf.to_crs(target_crs)
    liverpool_lsoa_geometries_gdf = lsoa_geometries_gdf[
        lsoa_geometries_gdf["LSOA21CD"].isin(liverpool_lsoas)
    ]

    if liverpool_lsoa_geometries_gdf.empty:
        raise ValueError("No LSOAs found for Liverpool in the provided LSOA shapefile.")

    return liverpool_lsoa_geometries_gdf


def plot_lsoa_geometries(
    liverpool_lsoa_geometries_gdf: gpd.GeoDataFrame, output_dir: str = OUTPUT_DIR
):
    """
    Plot Liverpool LSOA geometries and save as PNG and PDF.

    Args:
        liverpool_lsoa_geometries_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool LSOA geometries.
        output_dir (str): Output directory to save the plot image.
    """
    logging.info("Plotting LSOA geometries...")
    fig, ax = plt.subplots(figsize=(12, 6))
    liverpool_lsoa_geometries_gdf.plot(ax=ax)
    plt.title("Liverpool LSOA Geometries")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "liverpool_data_lsoas.png"))
    plt.savefig(os.path.join(output_dir, "liverpool_data_lsoas.pdf"))
    plt.show()


def merge_data(
    liverpool_hp_suitability_scores_pd: pd.DataFrame,
    liverpool_lsoa_geometries_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Merge LSOA suitability scores with geometries and convert to GeoDataFrame.

    Args:
        liverpool_hp_suitability_scores_pd (pd.DataFrame): DataFrame with Liverpool heat pump suitability scores.
        liverpool_lsoa_geometries_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool LSOA geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with merged suitability scores and geometries.

    Raises:
        ValueError: If the merged result is empty or missing geometry.
    """
    logging.info("Merging suitability data with geometries...")
    merged = liverpool_hp_suitability_scores_pd.merge(
        liverpool_lsoa_geometries_gdf, on="LSOA21CD", how="left"
    )

    # Validate merged data
    if merged.empty:
        raise ValueError("Merging resulted in an empty DataFrame. Check LSOA codes.")
    if merged["geometry"].isna().any():
        raise ValueError("Some records are missing geometry after merging.")

    liverpool_hp_suitability_gdf = gpd.GeoDataFrame(
        merged, geometry="geometry", crs=TARGET_CRS
    )
    return liverpool_hp_suitability_gdf


def plot_overlay(
    liverpool_hp_suitability_gdf: gpd.GeoDataFrame,
    liverpool_hnz_filepath: str = LIVERPOOL_GPKG_PATH,
    output_dir: str = OUTPUT_DIR,
):
    """
    Overlay of DESNZ pilot heat network zones on LSOAs in Liverpool. Saves the plot as PNG and PDF.

    Args:
        liverpool_hp_suitability_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool heat pump suitability scores and geometries.
        liverpool_hnz_filepath (str): Path to the Liverpool heat network zones GeoPackage.
        output_dir (str): Output directory to save the plot image.
    """
    logging.info("Plotting overlay of DESNZ pilot heat network zones...")
    gdf = pyogrio.read_dataframe(
        liverpool_hnz_filepath, layer="heat-network-zone-map-Liverpool"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, color="blue", alpha=0.5, label="DESNZ pilot heat network zones")
    liverpool_hp_suitability_gdf[
        liverpool_hp_suitability_gdf["DESNZ_pilot_fraction"] > 0.0
    ].plot(ax=ax, color="pink", alpha=0.5, label="LSOAs with DESNZ pilot heat network")
    plt.title("Overlay of DESNZ pilot heat network zones on Liverpool LSOAs")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "overlay_desnz_pilot.png"))
    plt.savefig(os.path.join(output_dir, "overlay_desnz_pilot.pdf"))
    plt.show()


def plot_absolute_error_map(
    hp_suitability_gdf: gpd.GeoDataFrame,
    score: float,
    title: str,
    filename: str,
    output_dir: str = OUTPUT_DIR,
):
    """
    Plot an absolute error map for areas in Liverpool based on the DESNZ pilot score. Saves the plot as PNG and PDF.

    Args:
        hp_suitability_gdf (gpd.GeoDataFrame): GeoDataFrame containing Liverpool's LSOA geometries and
                                               heat pump suitability scores, including 'absolute_error' and
                                               'DESNZ_pilot_fraction' columns.
        score (float): DESNZ pilot score threshold.
                       If > 0.0, plot regions where DESNZ_pilot_fraction >= score.
                       If 0, plot regions where DESNZ_pilot_fraction == 0.
                       Range from 0 to 1.
        title (str): Title for the plot.
        filename (str): Filename (PNG) to save the plot.
        output_dir (str): Output directory to save the plot image.

    Raises:
        ValueError: If score < 0.
    """
    logging.info(f"Plotting absolute error map for score threshold {score}...")
    if score < 0:
        raise ValueError("Score must be a non-negative value.")

    fig, ax = plt.subplots(figsize=(12, 6))

    if score > 0.0:
        subset = hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_fraction"] >= score]
        other = hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_fraction"] < score]
    else:  # score == 0
        subset = hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_fraction"] == 0]
        other = hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_fraction"] > 0]

    if subset.empty:
        logging.warning(f"No regions found for the given score threshold: {score}")

    subset.plot(column="absolute_error", ax=ax, legend=True, cmap="inferno")
    other.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5)

    # Update legend labeling for clarity
    cbar = ax.get_figure().get_axes()[1]
    cbar.set_ylabel(
        "Absolute Error (|Avg. HN Score - DESNZ HN Score|)", rotation=90, labelpad=15
    )

    plt.title(title)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.show()


def plot_hn_avg_score_vs_fraction_threshold(
    average_hn_scores_coverage_df: pl.DataFrame,
    fraction_thresholds_col: str,
    hn_avg_scores_col: str,
    filename: str,
    output_dir: str = OUTPUT_DIR,
):
    """
    Plot the average Nesta HN score against DESNZ Pilot Fraction Threshold, fit a regression line,
    and display R^2.

    Args:
        average_hn_scores_coverage_df (pl.DataFrame): DataFrame with average Nesta HN scores and DESNZ Pilot Fraction.
        fraction_thresholds_col (str): Column name for DESNZ Pilot Fraction Threshold.
        hn_avg_scores_col (str): Column name for Nesta Heat Network Avg Score Weighted.
    """
    logging.info("Plotting average HN score vs DESNZ Pilot Fraction Threshold...")
    fraction_thresholds = average_hn_scores_coverage_df[
        fraction_thresholds_col
    ].to_list()
    hn_avg_scores = average_hn_scores_coverage_df[hn_avg_scores_col].to_list()

    x = np.array(fraction_thresholds)
    y = np.array(hn_avg_scores)

    # Fit a linear model
    coefficients = np.polyfit(x, y, 1)
    model = np.poly1d(coefficients)
    predicted_y = model(x)
    r_squared = r2_score(y, predicted_y)

    y_min = min(hn_avg_scores) - 0.05
    y_max = max(hn_avg_scores) + 0.05

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        fraction_thresholds,
        hn_avg_scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Data",
    )
    ax.plot(
        fraction_thresholds,
        predicted_y,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Regression line",
    )

    ax.set_ylim(y_min, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Break mark on the y-axis
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax.set_title(
        "Average Nesta HN Score vs Fraction of DESNZ HN zone in LSOA", fontsize=14
    )
    ax.set_xlabel("Fraction of DESNZ HN zone in LSOA", fontsize=12)
    ax.set_ylabel("Average Nesta HN Score", fontsize=12)
    ax.grid(True)
    ax.text(
        0.05,
        0.95,
        f"$R^2 = {r_squared:.2f}$",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.show()


if __name__ == "__main__":
    # Argument parser for reading inputs from s3
    parser = argparse.ArgumentParser(
        description="Process heat network zones and calculate scores."
    )
    parser.add_argument(
        "--read_from_s3",
        type=bool,
        default=False,
        help="Read the input files from S3 bucket.",
    )
    args = parser.parse_args()
    read_from_s3 = args.read_from_s3

    (
        liverpool_hp_suitability_scores_pd,
        liverpool_lsoas,
        average_hn_scores_coverage_df,
    ) = load_data()
    liverpool_lsoa_geometries_gdf = preprocess_data(liverpool_lsoas)
    plot_lsoa_geometries(liverpool_lsoa_geometries_gdf)
    liverpool_hp_suitability_gdf = merge_data(
        liverpool_hp_suitability_scores_pd, liverpool_lsoa_geometries_gdf
    )
    plot_overlay(liverpool_hp_suitability_gdf)
    plot_absolute_error_map(
        liverpool_hp_suitability_gdf,
        score=0.000001,
        title="Absolute Error Map for when a DESNZ HN zone is present for Liverpool",
        filename="liverpool_data_with_pilot_score_1_v2",
    )
    plot_absolute_error_map(
        liverpool_hp_suitability_gdf,
        score=0,
        title="Absolute Error Map for when a DESNZ HN zone is absent for Liverpool",
        filename="liverpool_data_with_pilot_score_0_v2",
    )
    plot_hn_avg_score_vs_fraction_threshold(
        average_hn_scores_coverage_df,
        fraction_thresholds_col="DESNZ_pilot_fraction_threshold",
        hn_avg_scores_col="HN_N_avg_score_weighted",
        filename="HN_N_avg_score_vs_fraction_cover_by_DESNZ_HN",
    )
