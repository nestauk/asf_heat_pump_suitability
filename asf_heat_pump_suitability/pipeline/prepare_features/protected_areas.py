import geopandas as gpd
import pandas as pd
import numpy as np
import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import get_datasets


def load_transform_df_uprn_in_protected_area(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Generate dataframe of UPRNs located within building conservation areas in England and Wales and UPRNS
    located within Scottish World Heritage sites.

    Args:
        gdf (pl.DataFrame): dataframe with point geometries per UPRN in BNG

    Returns:
        pl.DataFrame: EPC UPRNs in building conservation areas in England and Wales and Scottish World Heritage Sites
    """
    ew_df = generate_df_uprn_in_cons_area(gdf).rename(
        {"in_conservation_area_ew": "in_protected_area"}
    )
    s_df = generate_df_uprn_in_whs(gdf).rename(
        {"in_world_heritage_site_s": "in_protected_area"}
    )

    return pl.concat([ew_df, s_df])


def generate_df_uprn_in_cons_area(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Generate dataframe of UPRNs located within building conservation areas in England and Wales.

    Args:
        gdf (pl.DataFrame): dataframe with point geometries per UPRN in BNG

    Returns:
        pl.DataFrame: EPC UPRNs in building conservation areas in England and Wales
    """
    cons_areas_gdf = transform_gdf_building_cons_areas()

    gdf = gdf.sjoin(
        cons_areas_gdf, how="inner", predicate="intersects"
    ).drop_duplicates(subset="UPRN")

    return pl.from_pandas(gdf[["UPRN", "in_conservation_area_ew"]])


def transform_gdf_building_cons_areas() -> gpd.GeoDataFrame:
    """
    Load, transform, and concatenate building conservation areas from England and Wales. Resulting GeoDataFrame is in
    CRS EPSG:27700 British National Grid.

    Returns:
        gpd.GeoDataFrame: building conservation areas in England and Wales
    """
    e_gdf = get_datasets.load_gdf_historic_england_conservation_areas(
        columns=["geometry"]
    ).to_crs("EPSG:27700")
    w_gdf = get_datasets.load_gdf_welsh_gov_conservation_areas(columns=["geometry"])

    gdf = pd.concat([e_gdf, w_gdf])
    gdf["in_conservation_area_ew"] = True

    return gdf


def generate_df_uprn_in_whs(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Generate dataframe to flag UPRNs located within World Heritage Sites in Scotland.

    Args:
        gdf (pl.DataFrame): dataframe with point geometries per UPRN in BNG

    Returns:
        pl.DataFrame: EPC UPRNs with flag for World Heritage Sites in Scotland
    """
    whs_gdf = load_transform_gdf_scottish_world_heritage_sites()

    gdf = gdf.sjoin(whs_gdf, how="left", predicate="intersects").drop_duplicates(
        subset="UPRN"
    )
    gdf["in_world_heritage_site_s"] = gdf["in_world_heritage_site_s"].fillna(False)

    return pl.from_pandas(gdf[["UPRN", "in_world_heritage_site_s"]])


def load_transform_gdf_scottish_world_heritage_sites() -> gpd.GeoDataFrame:
    """
    Load and transform Scottish World Heritage Sites geospatial data. CRS EPSG:27700, British National Grid.

    Returns:
        gpd.GeoDataFrame: Scottish World Heritage Sites
    """
    gdf = gpd.read_file(
        config["data_source"]["S_historic_environment_scotland_world_heritage_sites"],
        columns=["geometry"],
    )
    gdf["in_world_heritage_site_s"] = True

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
        [ladcd_col, "in_conservation_area_ew"]
    ].replace("No data available for publication by HE", np.nan)

    df = df.groupby(ladcd_col).agg({"in_conservation_area_ew": "count"})
    df["lad_conservation_area_data_available_ew"] = df[
        "in_conservation_area_ew"
    ].astype(bool)
    df = df.drop(columns=["in_conservation_area_ew"]).reset_index()

    return pl.from_pandas(df)
