import geopandas as gpd
import pandas as pd
import numpy as np
import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import epc


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
        columns=["geometry", "name"]
    ).to_crs("EPSG:27700")
    # Remove erroneous point where the entirety of Leicester is classified as a conservation zone
    e_gdf = e_gdf[e_gdf["name"] != "No data available for publication by HE"][
        ["geometry"]
    ].reset_index(drop=True)

    w_gdf = get_datasets.load_gdf_welsh_gov_conservation_areas(columns=["geometry"])

    gdf = pd.concat([e_gdf, w_gdf]).drop_duplicates(subset=["geometry"])
    gdf["in_conservation_area_ew"] = True

    return gdf


def generate_df_uprn_in_whs(
    gdf: gpd.GeoDataFrame, country_col: str = "COUNTRY"
) -> pl.DataFrame:
    """
    Generate dataframe to flag UPRNs located within World Heritage Sites in Scotland.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with point geometries per UPRN in BNG
        country_col (str): column containing country names

    Returns:
        pl.DataFrame: EPC UPRNs with flag for World Heritage Sites in Scotland
    """
    whs_gdf = load_transform_gdf_scottish_world_heritage_sites()

    gdf = gdf[gdf[country_col] == "Scotland"].copy()
    gdf = gdf.sjoin(whs_gdf, how="left", predicate="intersects").drop_duplicates(
        subset="UPRN"
    )
    # Fill null with False if row has geometry, otherwise leave as null
    gdf.loc[~gdf["geometry"].is_empty, "in_world_heritage_site_s"] = gdf.loc[
        ~gdf["geometry"].is_empty, "in_world_heritage_site_s"
    ].fillna(False)

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
    Generate dataframe of local authority districts (LADs) in England and Wales with indicator of building conservation area data
    availability.

    Args:
        ladcd_col (str): name of column in local authority district (LAD) boundaries file with LAD codes

    Returns:
        pl.DataFrame: building conservation area data availability per LAD in England and Wales
    """
    cons_areas_gdf = transform_gdf_building_cons_areas()
    council_bounds = get_datasets.load_gdf_ons_council_bounds()

    # Join conservation areas to their councils
    df = council_bounds.sjoin(cons_areas_gdf, how="left", predicate="intersects")[
        [ladcd_col, "in_conservation_area_ew"]
    ]

    df = df.groupby(ladcd_col).agg({"in_conservation_area_ew": "count"})
    df["lad_conservation_area_data_available_ew"] = df[
        "in_conservation_area_ew"
    ].astype(bool)
    df = df.drop(columns=["in_conservation_area_ew"]).reset_index()
    df = pl.from_pandas(df)

    df = (
        epc.extend_df_country_col(df, lsoa_col="LAD23CD")
        .filter(pl.col("country").is_in(["England", "Wales"]))
        .drop("country")
    )

    return df


def extend_df_protected_area_bool(
    features_df: pl.DataFrame,
    protected_areas_df: pd.DataFrame,
) -> pl.DataFrame:
    """
    Extend `features_df` with boolean column indicating whether each UPRN is within a protected area.

    Args:
        features_df (pl.DataFrame): Dataframe containing UPRNs to be labelled with protected area boolean flag.
        protected_areas_df (pd.DataFrame): Dataframe containing UPRNs and boolean flag indicating whether each UPRN is within a protected area.

    Returns:
        pl.DataFrame: Input dataframe extended with boolean column indicating whether each UPRN is within a protected area.
    """

    features_df = (
        features_df.join(
            protected_areas_df.select(["UPRN", "in_protected_area"]),
            how="left",
            on="UPRN",
        ).with_columns(pl.col("in_protected_area").fill_null(False))
        # .rename({"in_protected_area": "in_conservation_area"})
    )

    return features_df
