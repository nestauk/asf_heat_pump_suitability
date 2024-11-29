"""
Module for identifying and analyzing potential anchor properties in LSOAs.
This script can be run independentally and will output a CSV file with a list of LSOAs, the number of anchor properties in each LSOA, and the categories of anchor properties present.
"""

# TODO implement building footprint data for improved identification accuracy

import logging
from typing import Set
from pathlib import Path

import geopandas as gpd
import pandas as pd

from asf_heat_pump_suitability.getters.s3_getters import load_s3_data
from asf_heat_pump_suitability import config

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
INPUT_CRS = "EPSG:4326"
PROCESSING_CRS = "EPSG:27700"  # British National Grid

# Set of anchor property types
ANCHOR_PROPERTIES = {
    # TODO refine
    # alternate_categories for each type need to be further investigated to ensure that only the relevent building types are selected
    "hospital",
    "high_school",
    "university_housing",
    "college_university",
    "public_service_and_government",
    "central_government_office",
    "government_services",
    "local_and_state_government_offices",
    "sports_and_recreation_venue",
    "water_park",
    "ice_skating_rink",
    "swimming_pool",
    "jail_and_prison",
    "shopping_center",
    "airport_terminal",
    "airport",
    "superstore",
    "department_store",
}


def _safe_join_str_categories(categories: pd.Series) -> str:
    """
    Safely join category strings.

    Args:
        categories: Series of categories to join

    Returns:
        str: Comma-separated string of unique categories
    """
    valid_categories = set(str(cat) for cat in categories if pd.notna(cat))
    return ", ".join(sorted(valid_categories)) if valid_categories else ""


def load_gdf_and_process_poi() -> gpd.GeoDataFrame:
    """
    Load and process Points of Interest data.

    Returns:
        gpd.GeoDataFrame: Processed POI data containing only anchor properties

    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Loading POI data...")

    required_columns = [
        "id",
        "country",
        "main_category",
        "alternate_category",
        "geometry",
    ]
    poi = gpd.read_file(
        config["data_source"]["UK_poi_locations"], columns=required_columns
    ).to_crs(INPUT_CRS)

    # Filter and process
    poi = poi[poi.country == '"GB"'].copy()
    poi["main_category"] = poi["main_category"].apply(
        lambda x: str.strip(x, '"') if pd.notna(x) else None
    )

    # Filter anchor properties and reproject
    anchor_properties = poi[poi.main_category.isin(ANCHOR_PROPERTIES)].to_crs(
        PROCESSING_CRS
    )
    anchor_properties = anchor_properties.drop_duplicates(
        subset="geometry", keep="first"
    )

    logger.info(f"Found {len(anchor_properties)} potential anchor properties")
    logger.info(f"Output CRS: {PROCESSING_CRS}")
    return anchor_properties


def identify_anchor_properties_gdf() -> gpd.GeoDataFrame:
    """
    Identify and analyze anchor properties within LSOAs.

    Returns:
        gpd.GeoDataFrame: Summary of anchor properties by LSOA containing columns:
            - lsoa: Unique identifier for the LSOA
            - lsoa_name: Name of the LSOA
            - anchor_count: Number of anchor properties in the LSOA
            - building_categories: List of main categories present
            - building_subcategories: List of subcategories present
            - has_anchor_property: Boolean indicating presence of anchor properties
    """
    try:
        logger.info("Starting anchor property analysis...")

        # Load required data
        logger.info("Loading LSOA boundaries...")
        # Load and process LSOA boundary data
        lsoa_gdf = gpd.read_file(
            config["data_source"]["EW_lsoa_bounds"],
            columns=["LSOA21CD", "LSOA21NM", "geometry"],
        ).to_crs(PROCESSING_CRS)
        anchor_properties = load_gdf_and_process_poi()

        lsoa_with_anchors = gpd.sjoin(
            lsoa_gdf, anchor_properties, how="left", predicate="intersects"
        )

        lsoa_anchor_summary = (
            lsoa_with_anchors.groupby("LSOA21CD")
            .agg(
                {
                    "LSOA21NM": "first",
                    "id": "count",
                    "main_category": _safe_join_str_categories,
                    "alternate_category": _safe_join_str_categories,
                }
            )
            .reset_index()
        )

        lsoa_anchor_summary.columns = [
            "lsoa",
            "lsoa_name",
            "anchor_count",
            "building_categories",
            "building_subcategories",
        ]

        lsoa_anchor_summary["has_anchor_property"] = (
            lsoa_anchor_summary["anchor_count"] > 0
        )

        logger.info(
            f"Found {lsoa_anchor_summary['has_anchor_property'].sum()} LSOAs with suitable anchor properties"
        )

        return lsoa_anchor_summary

    except Exception as e:
        logger.error(f"Error in anchor property analysis: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        results = identify_anchor_properties_gdf()

        output_path = Path("outputs/reports/anchor_property_analysis.csv")
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
