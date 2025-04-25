"""
Utilities for plotting heat network zone (HNZ) suitability and related spatial data.
Includes functions to:
- Compute global min/max suitability scores.
- Generate scatter plots comparing ASHP vs. HN scores.
- Plot LSOA geometries and overlays with DESNZ pilot zones.
- Visualise absolute error maps and HN score trends.
Saves plots as PNG/PDF.
"""

import os
import logging
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Tuple
import matplotlib.patches as mpatches


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


def plot_ashp_vs_hn_scatter(
    merged_df: pd.DataFrame,
    la_name: str,
    global_x_min: float,
    global_x_max: float,
    global_y_min: float,
    global_y_max: float,
    output_dir: str,
) -> None:
    """
    Creates a scatter plot of ASHP (x-axis) vs Nesta HN (y-axis) scores for each LSOA,
    coloring each point by the DESNZ pilot fraction, with fixed axis ranges *across* all LAs.

    Args:
        merged_df (pd.DataFrame): DataFrame containing:
            - 'ASHP_N_avg_score_weighted' (float): ASHP suitability scores.
            - 'HN_N_avg_score_weighted' (float): HN suitability scores.
            - 'DESNZ_pilot_fraction' (float): Fraction of DESNZ pilot presence.

        la_name (str): Name of the local authority to include in the plot title.
        global_x_min (float): Minimum x-axis value (ASHP score) across all LAs.
        global_x_max (float): Maximum x-axis value (ASHP score) across all LAs.
        global_y_min (float): Minimum y-axis value (HN score) across all LAs.
        global_y_max (float): Maximum y-axis value (HN score) across all LAs.
        output_dir (str): Directory where the PNG and PDF files will be saved.
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

    out_png = os.path.join(output_dir, f"{la_name}_ashp_vs_hn_scatter_fixed.png")
    out_pdf = os.path.join(output_dir, f"{la_name}_ashp_vs_hn_scatter_fixed.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_lsoa_geometries(
    la_lsoa_geometries_gdf: gpd.GeoDataFrame,
    la_name: str,
    output_dir: str,
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

    plt.savefig(os.path.join(output_dir, f"{la_name}_lsoas.png"))
    plt.savefig(os.path.join(output_dir, f"{la_name}_lsoas.pdf"))
    plt.close(fig)


def plot_overlay(
    la_hp_suitability_gdf: gpd.GeoDataFrame,
    la_name: str,
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Overlays the LA's DESNZ pilot fraction > 0 on top of the LA's heat network zones from the GPKG file,
    if found. Saves to PNG and PDF.

    Args:
        la_hp_suitability_gdf (gpd.GeoDataFrame): GeoDataFrame with LA's LSOA geometries and 'DESNZ_pilot_fraction' > 0.
        la_name (str): Name of the local authority.
        input_dir (str): Directory to read the GPKG file from.
        output_dir (str): Directory to save plot outputs.
    """
    logging.info(f"Plotting overlay of DESNZ pilot heat network zones for {la_name}...")

    gpkg_filename = f"{la_name}_with_desnz_hn_lsoa.gpkg"
    gpkg_local_file_path = os.path.join(input_dir, gpkg_filename)

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

    out_png = os.path.join(output_dir, f"{la_name}_overlay_desnz_pilot.png")
    out_pdf = os.path.join(output_dir, f"{la_name}_overlay_desnz_pilot.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)


def plot_absolute_error_map(
    la_hp_suitability_gdf: gpd.GeoDataFrame,
    la_name: str,
    score: float,
    output_dir: str,
) -> None:
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
        cmap="cividis",
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

    out_png = os.path.join(output_dir, f"{la_name}_abs_error_map_score_{score}.png")
    out_pdf = os.path.join(output_dir, f"{la_name}_abs_error_map_score_{score}.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_hn_avg_score_vs_fraction_threshold(
    average_hn_scores_coverage_df: pl.DataFrame,
    la_name: str,
    output_dir: str,
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

    out_png = os.path.join(output_dir, f"{la_name}_hn_avg_score_vs_fraction.png")
    out_pdf = os.path.join(output_dir, f"{la_name}_hn_avg_score_vs_fraction.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)
