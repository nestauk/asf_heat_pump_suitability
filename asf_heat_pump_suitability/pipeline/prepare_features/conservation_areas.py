import geopandas as gpd
import pandas as pd
import numpy as np
import polars as pl
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import lat_lon


def transform_gdf_conservation_areas_england() -> gpd.GeoDataFrame:
    """
    Load and transform Historic England conservation areas to CRS EPSG:27700.

    Returns:
        gpd.GeoDataFrame: Historic England conservation areas in CRS EPSG:27700
    """
    gdf = get_datasets.load_gdf_historic_england_conservation_areas()[
        ["name", "geometry"]
    ].to_crs("EPSG:27700")

    return gdf


def generate_df_conservation_area_data_availability(
    ladcd_col: str = "LAD23CD",
) -> pl.DataFrame:
    """
    Generate dataframe of UK local authority districts (LADs) with indicator of conservation area data availability.

    Args:
        ladcd_col (str): name of column in local authority district (LAD) boundaries file with LAD codes

    Returns:
        pl.DataFrame: conservation area data availability per LAD in the UK
    """
    cons_areas_gdf = transform_gdf_conservation_areas_england()
    council_bounds = get_datasets.load_gdf_ons_council_bounds().to_crs(epsg="27700")

    # Join conservation areas to their councils
    df = council_bounds.sjoin(cons_areas_gdf, how="left", predicate="intersects")[
        [ladcd_col, "name"]
    ].replace("No data available for publication by HE", np.nan)

    df = df.groupby("LAD23CD").agg({"name": "count"})
    df["lad_conservation_area_data_available"] = df["name"].astype(bool)
    df = df.drop(columns=["name"])

    return pl.from_pandas(df)


def generate_df_uprn_to_cons_area(epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate dataframe of UPRNs located inside conservation areas.

    Args:
        epc_df (pl.DataFrame): EPC dataset with UPRN column and X and Y coordinate columns in BNG

    Returns:
        pl.DataFrame: dataframe of EPC UPRNs in conservation areas
    """
    # Convert BNG x, y coordinates to point geometries
    epc_df = lat_lon.generate_gdf_uprn_coords(epc_df)[["UPRN", "geometry"]]

    # Load conservation areas England
    cons_areas_gdf = transform_gdf_conservation_areas_england()

    # Join EPC UPRNs within or on boundaries of conservation areas
    epc_df = epc_df.sjoin(cons_areas_gdf, how="inner", predicate="intersects").drop(
        columns=["index_right", "geometry"]
    )

    # Fill unmatched UPRNs with NaN and case column as boolean
    epc_df["name"] = epc_df["name"].astype(bool)

    # Drop duplicate UPRNs introduced in cases where UPRN matched to multiple conservation areas
    epc_df = epc_df.drop_duplicates(subset="UPRN").rename(
        {"name": "in_conservation_area"}
    )

    return pl.from_pandas(epc_df)
