"""
This script processes and visualises DESNZ heat network (HN) and Nesta heat pump suitability data
for one or more Local Authorities (LAs). It iterates over LAs defined in a configuration file
(config.hnz_config).

**Steps**:
1. **Load and Preprocess Data**:
   - Reads Parquet, JSON, and (optionally) GeoPackage files for each LA.
   - Extracts unique LSOA geometries.
   - Ensures the LSOA shapefile is in the target CRS (e.g., EPSG:27700).
   - Filters LSOA geometries to just those in the current LA.

2. **Plot LA LSOA Geometries** and save to PNG/PDF.

3. **Merge Suitability Scores with Geometries**:
   - Convert to a GeoDataFrame.

4. **Plot Overlay of DESNZ Pilot Score > 0** on LA heat network zones (if a GPKG is found).

5. **Plot Absolute Error Maps**:
   - Visualises absolute error for LSOAs inside vs. outside DESNZ pilot zones.

6. **Plot Nesta HN Scores vs. DESNZ Pilot Fraction Threshold**:
   - Loads and plots average Nesta HN scores for different area coverage thresholds.
   - Fits a linear regression line and displays R².

**Usage**:
    python plot_comparison_of_hn_zones_all.py [--read_from_s3]

**Example**:
    # Run the script with local files only
    python plot_comparison_of_hn_zones_all.py

    # Run the script, reading files from S3
    python plot_comparison_of_hn_zones_all.py --read_from_s3
"""

import os
import json
import logging
import argparse
import boto3
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Tuple, List
import matplotlib.patches as mpatches
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability.getters.base_getters import get_df_from_parquet
from config.hnz_config import (
    LOCAL_AUTHORITIES,
    OUTPUT_DIR,  # "outputs/hn_zones/output_data/"
    S3_BUCKET,
    S3_KEY_DIR,
    NESTA_HPS_PARQUET_LOCAL,
    NESTA_HPS_PARQUET_S3,
)


####################
# 1. Configurations
####################
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
INPUT_DIR = OUTPUT_DIR  # your final data files are here
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_plots_all/")


# This is the path to the *national* LSOA shapefile (either local or S3).
# If your script can read from S3, handle that similarly to your existing code.
LSOA_SHP_PATH = (
    "s3://asf-heat-pump-suitability/source_data/"
    "Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/"
    "LSOA_2021_EW_BFE_V9.shp"
)

# Target CRS (ensure LSOA shapefiles and GPKGs are converted if they differ)
TARGET_CRS = "EPSG:27700"

###############
# NEW OR MODIFIED
###############


def load_HN_ASHP_scores(read_from_s3: bool = False) -> pd.DataFrame:
    """
    Loads the *full* Parquet containing ASHP and HN suitability scores (for *all* LAs).
    Returns a Pandas DataFrame with columns like:
      - 'lsoa'
      - 'ASHP_N_avg_score_weighted'
      - 'HN_N_avg_score_weighted'

    Args:
        read_from_s3 (bool): Whether to read the Parquet file from an s3 path or local path.

    Returns:
        pd.DataFrame
    """
    path = NESTA_HPS_PARQUET_S3 if read_from_s3 else NESTA_HPS_PARQUET_LOCAL
    logging.info(f"Loading global HP suitability data from {path}...")

    df_hps_polars = get_df_from_parquet(
        path, read_from_s3
    )  # function auto-detects "s3://" vs local
    df_hps = df_hps_polars.to_pandas()

    needed = {"lsoa", "ASHP_N_avg_score_weighted", "HN_N_avg_score_weighted"}
    if not needed.issubset(df_hps.columns):
        raise ValueError(f"Missing columns {needed} in: {df_hps.columns}")

    logging.info("Successfully loaded HN and ASHP scores.")
    return df_hps


def get_global_min_max(
    df_hp_suitability: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Computes global min/max for ASHP and HN suitability scores across *all LAs*.

    Args:
        df_hp_suitability (pd.DataFrame): DataFrame containing:
            ['ASHP_N_avg_score_weighted', 'HN_N_avg_score_weighted'] for *all* LAs.

    Returns:
        (x_min, x_max, y_min, y_max) with rounded values.
    """
    x_min = df_hp_suitability["ASHP_N_avg_score_weighted"].min()
    x_max = df_hp_suitability["ASHP_N_avg_score_weighted"].max()
    y_min = df_hp_suitability["HN_N_avg_score_weighted"].min()
    y_max = df_hp_suitability["HN_N_avg_score_weighted"].max()

    # Round for better readability
    x_min, x_max = round(x_min, 2), round(x_max, 2)
    y_min, y_max = round(y_min, 2), round(y_max, 2)

    return x_min, x_max, y_min, y_max


def plot_ashp_vs_hn_scatter_fixed_axes(
    merged_df: pd.DataFrame,
    la_name: str,
    global_x_min: float,
    global_x_max: float,
    global_y_min: float,
    global_y_max: float,
    output_dir: str = OUTPUT_PLOTS_DIR,
) -> None:
    """
    Creates a scatter plot of ASHP (x-axis) vs Nesta HN (y-axis) scores for each LSOA,
    coloring each point by the DESNZ pilot fraction, with fixed axis ranges *across* all LAs.

    merged_df must contain:
      - 'ASHP_N_avg_score_weighted'
      - 'HN_N_avg_score_weighted'
      - 'DESNZ_pilot_fraction'

    Saves a PNG and PDF to output_dir.
    """
    required_cols = [
        "ASHP_N_avg_score_weighted",
        "HN_N_avg_score_weighted",
        "DESNZ_pilot_fraction",
    ]
    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        logging.warning(f"Cannot plot scatter: missing columns {missing}")
        return

    x = merged_df["ASHP_N_avg_score_weighted"]
    y = merged_df["HN_N_avg_score_weighted"]
    c = merged_df["DESNZ_pilot_fraction"]

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        x, y, c=c, cmap="plasma", alpha=0.5, s=50, edgecolor="black", linewidth=0.3
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("DESNZ Pilot Fraction", rotation=270, labelpad=15)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Fix axis limits to global min/max
    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)

    ax.set_xlabel("ASHP (N) Weighted Score")
    ax.set_ylabel("Nesta HN (N) Weighted Score")
    ax.set_title(f"{la_name} – ASHP vs. Nesta HN Score\nColoured by DESNZ Fraction")

    ax.grid(True, linestyle="--", alpha=0.3)

    la_snake = la_name.lower().replace(" ", "_")
    out_png = os.path.join(output_dir, f"{la_snake}_ashp_vs_hn_scatter_fixed.png")
    out_pdf = os.path.join(output_dir, f"{la_snake}_ashp_vs_hn_scatter_fixed.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


########################################
# END of new scatter-plot-related functions
########################################


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


def plot_lsoa_geometries(
    la_lsoa_geometries_gdf: gpd.GeoDataFrame,
    la_name: str,
    output_dir: str = OUTPUT_PLOTS_DIR,
) -> None:
    """
    Plots the LSOA geometries for the specified local authority, saving as PNG and PDF.

    Args:
        la_lsoa_geometries_gdf (gpd.GeoDataFrame): GeoDataFrame of the LA's LSOA geometries.
        la_name (str): Name of the local authority.
        output_dir (str): Directory to save the output plots.
    """
    logging.info(f"Plotting LSOA geometries for {la_name}...")
    fig, ax = plt.subplots(figsize=(12, 6))
    la_lsoa_geometries_gdf.plot(ax=ax)
    plt.title(f"{la_name} - LSOA Geometries")
    plt.axis("off")

    la_snake = la_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{la_snake}_lsoas.png"))
    plt.savefig(os.path.join(output_dir, f"{la_snake}_lsoas.pdf"))
    plt.close(fig)


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


def plot_overlay(
    la_hp_suitability_gdf: gpd.GeoDataFrame,
    la_name: str,
    output_dir: str = OUTPUT_PLOTS_DIR,
):
    """
    Overlays the LA's DESNZ pilot fraction > 0 on top of the LA's heat network zones from the GPKG file,
    if found. Saves to PNG and PDF.

    Args:
        la_hp_suitability_gdf (gpd.GeoDataFrame): GeoDataFrame with LA's LSOA geometries and 'DESNZ_pilot_fraction' > 0.
        la_name (str): Name of the local authority.
        output_dir (str): Directory to save plot outputs.
    """
    logging.info(f"Plotting overlay of DESNZ pilot heat network zones for {la_name}...")
    la_snake = la_name.lower().replace(" ", "_")
    gpkg_filename = f"{la_snake}_with_desnz_hn_lsoa.gpkg"
    gpkg_local_file_path = os.path.join(INPUT_DIR, gpkg_filename)

    # Split LSOAs into "in zone" vs. "out of zone"
    in_zone = la_hp_suitability_gdf[la_hp_suitability_gdf["DESNZ_pilot_fraction"] > 0]
    out_zone = la_hp_suitability_gdf[la_hp_suitability_gdf["DESNZ_pilot_fraction"] == 0]
    if not os.path.exists(gpkg_local_file_path):
        logging.warning(
            f"No GPKG file found for {la_name} at {gpkg_local_file_path}. Skipping overlay."
        )
        return

    try:
        hn_gdf = pyogrio.read_dataframe(gpkg_local_file_path)
    except Exception as e:
        logging.error(f"Failed to read GPKG for {la_name}: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    # 1) Out-of-zone LSOAs (fraction=0)
    out_zone.plot(
        ax=ax,
        edgecolor="black",
        facecolor="white",
        linewidth=0.1,
        label="LSOAs with no HN zone presence",
    )
    # 2) Heat network zones from GPKG
    hn_gdf.plot(ax=ax, color="blue", alpha=0.5, label="DESNZ pilot HN zones")
    in_zone.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.1,
        color="pink",
        alpha=0.5,
        label="LSOAs with presence of HN zone",
    )
    # Create custom legend handles:
    out_patch = mpatches.Patch(
        edgecolor="black",
        facecolor="white",
        linewidth=0.5,
        label="No DESNZ pilot zone (fraction=0)",
    )
    blue_patch = mpatches.Patch(color="blue", alpha=0.5, label="DESNZ pilot HN zones")
    pink_patch = mpatches.Patch(
        color="pink", alpha=0.5, label="LSOAs with presence of HN zone"
    )

    # Add the legend
    ax.legend(
        handles=[out_patch, blue_patch, pink_patch],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
    )
    plt.title(f"{la_name} - Overlay of DESNZ Pilot Heat Network Zones")
    plt.axis("off")

    out_png = os.path.join(output_dir, f"{la_snake}_overlay_desnz_pilot.png")
    out_pdf = os.path.join(output_dir, f"{la_snake}_overlay_desnz_pilot.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)


def plot_absolute_error_map(
    la_hp_suitability_gdf: gpd.GeoDataFrame,
    la_name: str,
    score: float,
    output_dir: str = OUTPUT_PLOTS_DIR,
):
    """
    Plots an absolute error map for the specified local authority based on DESNZ pilot fraction.
    Saves the plot as PNG and PDF.

    Args:
        la_hp_suitability_gdf (gpd.GeoDataFrame): LA's LSOA geometries + 'absolute_error' & 'DESNZ_pilot_fraction'.
        la_name (str): Name of the local authority (for labeling).
        score (float): Score threshold. If > 0, filter 'DESNZ_pilot_fraction' >= score. If 0, fraction == 0 only.
        output_dir (str): Directory where plots will be saved.

    Raises:
        ValueError: If 'score' is negative.

    if score < 0:
        raise ValueError("Score must be a non-negative value.")
    """

    if score < 0:
        raise ValueError("Score must be a non-negative value.")

    logging.info(f"Plotting absolute error map for {la_name} with threshold {score}...")

    # Separate the subset vs. other
    if score > 0:
        subset = la_hp_suitability_gdf[
            la_hp_suitability_gdf["DESNZ_pilot_fraction"] >= score
        ]
        other = la_hp_suitability_gdf[
            la_hp_suitability_gdf["DESNZ_pilot_fraction"] < score
        ]
        label_str = "DESNZ pilot fraction > 0"
    else:
        subset = la_hp_suitability_gdf[
            la_hp_suitability_gdf["DESNZ_pilot_fraction"] == 0
        ]
        other = la_hp_suitability_gdf[la_hp_suitability_gdf["DESNZ_pilot_fraction"] > 0]
        label_str = "DESNZ pilot fraction = 0"

    # 1) Compute a global min/max for the entire LA:
    global_min = la_hp_suitability_gdf["absolute_error"].min()
    global_max = la_hp_suitability_gdf["absolute_error"].max()

    fig, ax = plt.subplots(figsize=(12, 6))

    # 2) Pass vmin, vmax so the color scale is fixed:
    subset.plot(
        column="absolute_error",
        ax=ax,
        legend=True,
        cmap="RdYlGn_r",
        vmin=global_min,
        vmax=global_max,
        edgecolor="black",
        linewidth=0.2,
    )

    # 3) Plot the 'other' subset as an outline
    other.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.2)

    cbar = ax.get_figure().get_axes()[1]  # colorbar axis
    cbar.set_ylabel(
        "Absolute Error (|Avg. HN Score - DESNZ HN Score|)\nLower = Better",
        rotation=90,
        labelpad=15,
    )

    plt.title(f"{la_name} - Absolute Error Map ({label_str})")
    plt.axis("off")

    la_snake = la_name.lower().replace(" ", "_")
    out_png = os.path.join(output_dir, f"{la_snake}_abs_error_map_score_{score}.png")
    out_pdf = os.path.join(output_dir, f"{la_snake}_abs_error_map_score_{score}.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_hn_avg_score_vs_fraction_threshold(
    average_hn_scores_coverage_df: pl.DataFrame,
    la_name: str,
    output_dir: str = OUTPUT_PLOTS_DIR,
):
    """
    Plots the average Nesta HN score against the DESNZ Pilot Fraction threshold for a given LA.
    Fits a linear regression line and displays R² in the top-left corner of the plot.
    Saves plot as PNG/PDF.

    Args:
        average_hn_scores_coverage_df (pl.DataFrame): Data with columns:
            ['DESNZ_pilot_fraction_threshold', 'HN_N_avg_score_weighted'].
        la_name (str): Local Authority name for labeling.
        output_dir (str): Directory to save the plot.
    """
    logging.info(
        f"Plotting average HN score vs DESNZ pilot fraction threshold for {la_name}..."
    )
    fraction_thresholds = average_hn_scores_coverage_df[
        "DESNZ_pilot_fraction_threshold"
    ].to_list()
    hn_avg_scores = average_hn_scores_coverage_df["HN_N_avg_score_weighted"].to_list()

    x = np.array(fraction_thresholds)
    y = np.array(hn_avg_scores)

    if len(x) == 0:
        logging.warning(f"No threshold data to plot for {la_name}.")
        return

    # Fit a linear model
    coeffs = np.polyfit(x, y, 1)
    model = np.poly1d(coeffs)
    predicted_y = model(x)
    r2 = r2_score(y, predicted_y)

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
        f"{la_name} - Average Nesta HN Score vs. Fraction of DESNZ HN zone in LSOA"
    )
    ax.set_xlabel("Fraction of DESNZ HN zone in LSOA", fontsize=12)
    ax.set_ylabel("Average Nesta HN Score", fontsize=12)
    ax.grid(True)

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.legend(frameon=False)

    la_snake = la_name.lower().replace(" ", "_")
    out_png = os.path.join(output_dir, f"{la_snake}_hn_avg_score_vs_fraction.png")
    out_pdf = os.path.join(output_dir, f"{la_snake}_hn_avg_score_vs_fraction.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparisons of heat network zones for all LAs."
    )
    parser.add_argument(
        "--read_from_s3",
        action="store_true",
        help="Read the input files from S3 rather than local disk.",
    )
    args = parser.parse_args()
    read_from_s3 = args.read_from_s3

    ### NEW OR MODIFIED ###
    # 1. Load the *full* HP suitability dataset once for all LAs (used for global min/max + ASHP column)
    df_hp_suitability_global = load_HN_ASHP_scores(read_from_s3=read_from_s3)

    # 2. Compute global min/max (ASHP vs. HN) from the full dataset
    global_x_min, global_x_max, global_y_min, global_y_max = get_global_min_max(
        df_hp_suitability_global
    )

    # Also extract just the ASHP column for merging onto each LA
    df_ashp_suitability = df_hp_suitability_global[
        ["lsoa", "ASHP_N_avg_score_weighted"]
    ]
    # (We assume local LA data already have 'HN_N_avg_score_weighted' or similar.)

    # Iterate over each Local Authority in config
    for la_name, la_value in LOCAL_AUTHORITIES.items():
        # If the LA is actually a region dict (e.g., Greater Manchester), skip or handle sub-LAs separately
        if isinstance(la_value, dict):
            logging.info(f"Skipping region '{la_name}' because it has sub-LAs.")
            continue

        logging.info(f"=== Processing {la_name} ===")

        try:
            # 1. Load data (HP Parquet, LSOA JSON, average threshold Parquet)
            hp_scores_pd, la_lsoas, avg_hn_scores_df = load_la_data(
                la_name, read_from_s3
            )

            # 2. Preprocess LSOA geometries
            la_lsoa_gdf = preprocess_data(la_lsoas)

            # 3. Plot LSOA geometries
            plot_lsoa_geometries(la_lsoa_gdf, la_name)

            # 4. Merge data
            la_hp_gdf = merge_data(hp_scores_pd, la_lsoa_gdf, la_name)

            # 5. Overlay of DESNZ pilot fraction zones
            plot_overlay(la_hp_gdf, la_name)

            # 6. Plot absolute error maps (example thresholds)
            plot_absolute_error_map(la_hp_gdf, la_name, score=0.000001)
            plot_absolute_error_map(la_hp_gdf, la_name, score=0)

            # 7. Plot average Nesta HN score vs. fraction coverage threshold
            plot_hn_avg_score_vs_fraction_threshold(avg_hn_scores_df, la_name)

            # 5. Merge the global ASHP column (df_ashp_suitability) onto the LA DataFrame
            la_hp_gdf_merged = la_hp_gdf.merge(
                df_ashp_suitability,
                how="left",
                left_on="LSOA21CD",  # from LA parquet
                right_on="lsoa",  # from the global CSV
            ).drop(
                columns=["lsoa"]
            )  # we only needed it for the join

            # 9. (New) Plot ASHP vs. HN scatter with *fixed* global axes
            plot_ashp_vs_hn_scatter_fixed_axes(
                merged_df=la_hp_gdf_merged,
                la_name=la_name,
                global_x_min=global_x_min,
                global_x_max=global_x_max,
                global_y_min=global_y_min,
                global_y_max=global_y_max,
                output_dir=OUTPUT_PLOTS_DIR,
            )

        except Exception as err:
            logging.error(f"Error processing {la_name}: {err}")

        logging.info(f"=== Finished {la_name} ===\n")
