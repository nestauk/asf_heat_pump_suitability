import geopandas as gpd
import pandas as pd
import numpy as np
import polars as pl
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import lat_lon


def transform_gdf_building_cons_areas() -> gpd.GeoDataFrame:
    """
    Load, transform, and concatenate building conservation areas from England and Wales. Resulting GeoDataFrame is in
    CRS EPSG:27700 British National Grid.

    Returns:
        gpd.GeoDataFrame: building conservation areas in England and Wales
    """
    e_gdf = (
        get_datasets.load_gdf_historic_england_conservation_areas(
            columns=["name", "geometry"]
        )
        .to_crs("EPSG:27700")
        .rename(columns={"name": "sitename"})
    )
    w_gdf = get_datasets.load_gdf_welsh_gov_conservation_areas(
        columns=["sitename", "geometry"]
    )

    gdf = pd.concat([e_gdf, w_gdf])

    return gdf


def generate_df_conservation_area_data_availability(
    ladcd_col: str = "LAD23CD",
) -> pl.DataFrame:
    """
    Generate dataframe of UK local authority districts (LADs) with indicator of building conservation area data
    availability.

    Args:
        ladcd_col (str): name of column in local authority district (LAD) boundaries file with LAD codes

    Returns:
        pl.DataFrame: building conservation area data availability per LAD in the UK
    """
    cons_areas_gdf = transform_gdf_building_cons_areas()
    council_bounds = get_datasets.load_gdf_ons_council_bounds()

    # Join conservation areas to their councils
    df = council_bounds.sjoin(cons_areas_gdf, how="left", predicate="intersects")[
        [ladcd_col, "sitename"]
    ].replace("No data available for publication by HE", np.nan)

    df = df.groupby(ladcd_col).agg({"sitename": "count"})
    df["lad_conservation_area_data_available"] = df["sitename"].astype(bool)
    df = df.drop(columns=["sitename"]).reset_index()

    return pl.from_pandas(df)


def generate_df_uprn_to_cons_area(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate dataframe of UPRNs located within building conservation areas in England and Wales.

    Args:
        df (pl.DataFrame): EPC dataset with UPRN column and X and Y coordinate columns in BNG

    Returns:
        pl.DataFrame: dataframe of EPC UPRNs in building conservation areas in England and Wales
    """
    # Convert BNG x, y coordinates to point geometries
    df = lat_lon.generate_gdf_uprn_coords(df)[["UPRN", "geometry"]]

    # Load and transform conservation areas in England and Wales
    cons_areas_gdf = transform_gdf_building_cons_areas()

    # Join EPC UPRNs within or on boundaries of conservation areas
    df = (
        df.sjoin(cons_areas_gdf, how="inner", predicate="intersects")
        .drop(columns=["index_right", "geometry"])
        .drop_duplicates(subset="UPRN")
    )

    # Set column as boolean
    df["sitename"] = df["sitename"].astype(bool)
    df = df.rename(columns={"sitename": "in_conservation_area"})

    return pl.from_pandas(df)
