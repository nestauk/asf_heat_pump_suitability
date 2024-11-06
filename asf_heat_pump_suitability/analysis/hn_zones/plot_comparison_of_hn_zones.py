"""
This script processes and visualizes heat network (HN) and heat pump suitability data for Liverpool.

Steps:
1. Load and preprocess data:
   - Read Parquet and GeoPackage files.
   - Extract unique LSOA geometries.
   - Load LSOA geometries and convert CRS if needed.
   - Filter Liverpool LSOAs.

2. Plot Liverpool LSOA geometries and save as PNG.

3. Merge suitability scores with geometries and convert to GeoDataFrame.

4. Plot and save overlay of DESNZ Pilot Score = 1 on Liverpool heat network zones.

5. Define and use a function to plot absolute error maps by DESNZ Pilot Score, saving the plots as PNG files.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pyogrio
import json
import os
import logging
from asf_heat_pump_suitability import PROJECT_DIR

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define constants for file paths
INPUT_DIR = os.path.join(
    PROJECT_DIR, "asf_heat_pump_suitability/analysis/hn_zones/output_data/"
)
LIVERPOOL_HP_SUITABILITY_PARQUET = os.path.join(
    INPUT_DIR, "liverpool_hp_suitability_scores_with_desnz.parquet"
)
LSOA_SHP_PATH = "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"
LIVERPOOL_GPKG_PATH = "s3://asf-heat-pump-suitability/heat_network_desnz_data/heat-network-zone-map-Liverpool.gpkg"
LSOA_JSON_PATH = os.path.join(INPUT_DIR, "liverpool_hp_suitability_lsoas.json")
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "asf_heat_pump_suitability/analysis/hn_zones/output_plots/"
)

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """
    Load the necessary data files.

    Returns:
        Tuple containing:
            - pd.DataFrame: DataFrame with Liverpool heat pump suitability scores.
            - List[str]: List of LSOA codes for Liverpool.
    """
    try:
        liverpool_hp_suitability_scores_pd = pd.read_parquet(
            LIVERPOOL_HP_SUITABILITY_PARQUET
        )
        with open(LSOA_JSON_PATH, "r") as file:
            liverpool_hp_suitability_lsoas = json.load(file)
        return liverpool_hp_suitability_scores_pd, liverpool_hp_suitability_lsoas
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(
    liverpool_hp_suitability_lsoas: list, lsoa_shp_path: str = LSOA_SHP_PATH
) -> gpd.GeoDataFrame:
    """
    Preprocess the data by extracting unique LSOA geometries and filtering Liverpool LSOAs.

    Args:
        liverpool_hp_suitability_lsoas (list): List of LSOA codes for Liverpool.
        lsoa_shp_path (str): Path to the LSOA shapefile.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filtered Liverpool LSOA geometries.
    """
    lsoa_geometries_gdf = gpd.read_file(lsoa_shp_path)
    if lsoa_geometries_gdf.crs != "EPSG:27700":  # Assuming British National Grid
        logging.info("Converting to EPSG:27700")
        lsoa_geometries_gdf = lsoa_geometries_gdf.to_crs("EPSG:27700")
    liverpool_lsoa_geometries_gdf = lsoa_geometries_gdf[
        lsoa_geometries_gdf["LSOA21CD"].isin(liverpool_hp_suitability_lsoas)
    ]
    return liverpool_lsoa_geometries_gdf


def plot_lsoa_geometries(liverpool_lsoa_geometries_gdf, output_dir: str = OUTPUT_DIR):
    """
    Plot Liverpool LSOA geometries and save as PNG.

    Args:
        liverpool_lsoa_geometries_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool LSOA geometries.
        output_dir (str): Output directory to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    liverpool_lsoa_geometries_gdf.plot(ax=ax)
    plt.title("Liverpool LSOA Geometries")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "liverpool_data_lsoas.png"))
    plt.show()


def merge_data(
    liverpool_hp_suitability_scores_pd: pd.DataFrame,
    liverpool_lsoa_geometries_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Merge suitability scores with geometries and convert to GeoDataFrame.

    Args:
        liverpool_hp_suitability_scores_pd (pd.DataFrame): DataFrame with Liverpool heat pump suitability scores.
        liverpool_lsoa_geometries_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool LSOA geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with merged suitability scores and geometries.
    """
    liverpool_hp_suitability_with_geometry = liverpool_hp_suitability_scores_pd.merge(
        liverpool_lsoa_geometries_gdf, on="LSOA21CD", how="left"
    )
    liverpool_hp_suitability_gdf = gpd.GeoDataFrame(
        liverpool_hp_suitability_with_geometry, geometry="geometry"
    )
    return liverpool_hp_suitability_gdf


def plot_overlay(
    liverpool_hp_suitability_gdf: gpd.GeoDataFrame,
    liverpool_heat_network_zones_filepath: str = LIVERPOOL_GPKG_PATH,
    output_dir: str = OUTPUT_DIR,
):
    """
    Plot and save overlay of DESNZ Pilot Score = 1 on Liverpool heat network zones.

    Args:
        liverpool_hp_suitability_gdf (gpd.GeoDataFrame): GeoDataFrame with Liverpool heat pump suitability scores and geometries.
        liverpool_heat_network_zones_filepath (str): Path to the Liverpool heat network zones GeoPackage.
        output_dir (str): Output directory to save the plot image.
    """
    gdf = pyogrio.read_dataframe(
        liverpool_heat_network_zones_filepath, layer="heat-network-zone-map-Liverpool"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, color="blue", alpha=0.5)
    liverpool_hp_suitability_gdf[
        liverpool_hp_suitability_gdf["DESNZ_pilot_score"] == 1
    ].plot(ax=ax, color="pink", alpha=0.5)
    plt.title("Overlay of DESNZ Pilot Score = 1 on Liverpool Heat Network Zones")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "overlay_desnz_pilot.png"))
    plt.show()


def plot_absolute_error_map(
    hp_suitability_gdf: gpd.GeoDataFrame,
    score: int,
    title: str,
    filename: str,
    output_dir: str = OUTPUT_DIR,
):
    """
    Plot an absolute error map for areas in Liverpool based on the DESNZ pilot score,
    overlaying regions with a different score as boundaries.

    Args:
        df (gpd.GeoDataFrame): GeoDataFrame containing Liverpool's LSOA geometries and
                               heat pump suitability scores, including absolute error values.
        score (int): DESNZ pilot score value for which to highlight areas.
                     Only regions matching this score will be color-filled.
        title (str): Title for the plot.
        filename (str): Path to save the plot image as a PNG file.
        output_dir (str): Output directory to save the plot image.

    This function:
        - Plots regions where DESNZ pilot score matches the specified `score`,
          coloring them by 'absolute_error' values.
        - Overlays boundaries of areas where DESNZ pilot score does not match the specified `score`.
        - Displays a legend and saves the plot as a PNG file.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_score"] == score].plot(
        column="absolute_error", ax=ax, legend=True, cmap="inferno"
    )
    hp_suitability_gdf[hp_suitability_gdf["DESNZ_pilot_score"] != score].plot(
        ax=ax, edgecolor="black", facecolor="none", linewidth=0.5
    )
    plt.title(title)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def main():
    """
    Main function to load, preprocess, and visualise data.
    """
    liverpool_hp_suitability_scores_pd, liverpool_hp_suitability_lsoas = load_data()
    liverpool_lsoa_geometries_gdf = preprocess_data(liverpool_hp_suitability_lsoas)
    plot_lsoa_geometries(liverpool_lsoa_geometries_gdf)
    liverpool_hp_suitability_gdf = merge_data(
        liverpool_hp_suitability_scores_pd, liverpool_lsoa_geometries_gdf
    )
    plot_overlay(liverpool_hp_suitability_gdf)
    plot_absolute_error_map(
        liverpool_hp_suitability_gdf,
        score=1,
        title="Absolute Error Map for DESNZ Pilot Score = 1 for Liverpool",
        filename="liverpool_data_with_pilot_score_1_v2.png",
    )
    plot_absolute_error_map(
        liverpool_hp_suitability_gdf,
        score=0,
        title="Absolute Error Map for DESNZ Pilot Score = 0 for Liverpool",
        filename="liverpool_data_with_pilot_score_0_v2.png",
    )


if __name__ == "__main__":
    main()
