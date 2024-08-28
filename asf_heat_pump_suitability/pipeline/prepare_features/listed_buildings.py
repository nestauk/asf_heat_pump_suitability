import polars as pl
import geopandas as gpd
import pandas as pd
from asf_heat_pump_suitability.getters import get_datasets
from tqdm import tqdm
import logging
from typing import List


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


def transform_gdf_listed_buildings(nation: str) -> gpd.GeoDataFrame:
    """
    Load and transform listed buildings polygons dataset for specified nation.

    Args:
        nation (str): UK nation to load listed buildings data for. Options: "England"; "Wales".

    Returns:
        gpd.GeoDataFrame: listed buildings dataset for specified nation with grade and geometry columns.
    """
    gdf = get_datasets.load_gdf_listed_buildings(nation, columns=["Grade", "geometry"])
    gdf = gdf.drop_duplicates(subset="geometry").rename(
        columns={"Grade": "listed_building_grade"}
    )

    return gdf


def sjoin_df_epc_with_listed_buildings(
    epc_df: pl.DataFrame,
    listed_buildings_gdf: gpd.GeoDataFrame,
    chunk_size: int = 100000,
) -> pl.DataFrame:
    """
    Spatial join EPC UPRNs with listed buildings polygons to return dataframe of EPC UPRNs which are located in listed
    buildings, along with the building grade.
    Args:
        epc_df (pl.DataFrame): Enhanced EPC dataset with X and Y coordinates
        listed_buildings_gdf (gpd.GeoDataFrame): listed buildings polygons dataset
        chunk_size (int): number of EPC rows in each partition. Default 100,000
    Returns:
        pl.DataFrame: EPC UPRNs with listed buildings grade
    """
    partitions = (
        epc_df.select(["UPRN", "X_COORDINATE", "Y_COORDINATE"])
        .with_row_index("chunk_id")
        .select(pl.col("chunk_id") // chunk_size)
    )

    dfs = []

    # Group based on the created index
    data_partitioned = epc_df.with_columns(partitions).partition_by("chunk_id")
    logging.info(f"Adding listed buildings to EPC in {len(data_partitioned)} chunks")
    for epc_chunk in tqdm(data_partitioned):
        epc_gdf = transform_df_EPC_X_and_Y_to_point(epc_chunk)[["UPRN", "geometry"]]
        df = epc_gdf.sjoin(listed_buildings_gdf, how="inner", predicate="intersects")[
            ["UPRN", "listed_building_grade"]
        ].drop_duplicates(subset="UPRN")

        dfs.append(df)

    df = pl.from_pandas(pd.concat(dfs))

    return df
