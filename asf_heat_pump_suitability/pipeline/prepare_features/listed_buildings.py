import polars as pl
import geopandas as gpd
import pandas as pd
from asf_heat_pump_suitability.getters.get_datasets import (
    get_df_historicengland_listedbuildings,
)


def transform_df_EPC_X_and_Y_to_point(
    enhanced_epc_df: pl.DataFrame,
    x_col: str = "X_COORDINATE",
    y_col: str = "Y_COORDINATE",
) -> gpd.GeoDataFrame:
    """
    Transform 'X' and 'Y' coordinates in EPC dataset to be in 'POINT' format.

    Args:
        df (pl.DataFrame): Enhanced EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with a column of 'POINT' (transformed from 'X' and 'Y' coordinate).
    """
    enhanced_epc_df = enhanced_epc_df.to_pandas()
    gdf = gpd.GeoDataFrame(
        enhanced_epc_df,
        geometry=gpd.points_from_xy(enhanced_epc_df[x_col], enhanced_epc_df[y_col]),
        crs="EPSG:27700",
    )
    return gdf


def get_filtered_df_listed_buildings() -> gpd.GeoDataFrame:
    """
    Filter out listed buildings from Historic England dataset.

    Returns:
        gpd.GeoDataFrame: Filtered Historic England dataset with only listed buildings grade and geometry.
    """
    # Get the Historic England dataset
    historic_england_gdf = get_df_historicengland_listedbuildings()
    # Filter out listed buildings
    relevant_columns = ["geometry", "Grade"]
    filtered_historic_england_gdf = historic_england_gdf[relevant_columns]

    return filtered_historic_england_gdf


def spatial_join_epc_with_listed_buildings(
    enhanced_epc_df: pl.DataFrame, listed_buildings_df: gpd.GeoDataFrame
) -> pl.DataFrame:
    """
    Spatial join EPC dataset with listed buildings dataset.
    Args:
        enhanced_epc_df (pl.DataFrame): Enhanced EPC dataset
        listed_buildings_df (gpd.GeoDataFrame): Filtered Historic England dataset with only listed buildings grade and geometry.
    Returns:
        gpd.GeoDataFrame: EPC dataset with listed buildings grade without geometry.
    """
    epc_gdf = transform_df_EPC_X_and_Y_to_point(enhanced_epc_df)
    epc_gdf_temp = epc_gdf[["geometry"]].copy()
    epc_gdf_temp["index"] = epc_gdf.index
    joined_gdf = gpd.sjoin(
        epc_gdf_temp, listed_buildings_df, how="left", predicate="intersects"
    )
    # Drop the geometry column
    joined_gdf = joined_gdf.drop(columns=["geometry", "index_right"])
    epc_gdf["index"] = epc_gdf.index
    result_gdf = epc_gdf.merge(
        joined_gdf[["Grade", "index"]], on="index", how="left"
    ).drop(columns=["geometry", "index"])
    return result_gdf


def convert_gpd_to_polars(gpd: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Convert GeoDataFrame to Polars DataFrame.

    Args:
        gpd (gpd.GeoDataFrame): GeoDataFrame to convert

    Returns:
        pl.DataFrame: Polars DataFrame
    """
    pdf = pd.DataFrame(gpd)
    pl_df = pl.from_pandas(pdf)
    return pl_df
