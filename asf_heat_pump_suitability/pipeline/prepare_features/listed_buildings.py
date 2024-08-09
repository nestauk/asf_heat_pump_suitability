import polars as pl
import geopandas as gpd
import pandas as pd
from asf_heat_pump_suitability.getters.get_datasets import get_df_listedbuildings
from tqdm import tqdm
import logging


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


def get_filtered_df_listed_buildings(data_source_key: str) -> gpd.GeoDataFrame:
    """
    Get and filter out listed buildings from Historic England dataset.

    Returns:
        gpd.GeoDataFrame: Filtered Historic England dataset with only listed buildings grade and geometry.
    """
    listed_building_gdf = get_df_listedbuildings(data_source_key)
    relevant_columns = ["geometry", "Grade"]
    filtered_listed_building_gdf = listed_building_gdf[relevant_columns]

    return filtered_listed_building_gdf


def spatial_join_epc_with_listed_buildings(
    enhanced_epc_df: pl.DataFrame,
    listed_buildings_df: gpd.GeoDataFrame,
    chunk_size: int = 100000,
) -> pl.DataFrame:
    """
    Spatial join EPC dataset with listed buildings dataset.
    Args:
        enhanced_epc_df (pl.DataFrame): Enhanced EPC dataset
        listed_buildings_df (gpd.GeoDataFrame): Filtered Historic England dataset with only listed buildings grade and geometry.
    Returns:
        pl.DataFrame: EPC dataset with listed buildings grade without geometry.
    """

    # Add a unique index column for merging later
    enhanced_epc_df = enhanced_epc_df.with_row_index("index")
    partitions = enhanced_epc_df.with_row_count("chunk_id").select(
        pl.col("chunk_id") // chunk_size
    )
    # group based on the created index, resulting chunk_size partitons
    data_partitioned = enhanced_epc_df.with_columns(partitions).partition_by("chunk_id")
    logging.info(f"Adding listed buildings to EPC in {len(data_partitioned)} chunks")
    for i, enhanced_epc_df_chunk in tqdm(enumerate(data_partitioned)):
        epc_gdf = transform_df_EPC_X_and_Y_to_point(enhanced_epc_df_chunk)
        epc_gdf_temp = epc_gdf[["geometry", "index"]].copy()
        joined_gdf_chunk = gpd.sjoin(
            epc_gdf_temp, listed_buildings_df, how="inner", predicate="intersects"
        )
        # Drop the geometry column
        joined_gdf_chunk = joined_gdf_chunk.drop(columns=["geometry", "index_right"])
        if i == 0:
            joined_gdf = joined_gdf_chunk
        else:
            joined_gdf = pd.concat([joined_gdf, joined_gdf_chunk])

    enhanced_epc_df = enhanced_epc_df.join(
        pl.from_pandas(joined_gdf), how="left", on="index"
    )

    return enhanced_epc_df


def merge_listed_buildings_nations(
    enhanced_epc_nationone_listed_buildings_df: pl.DataFrame,
    enhanced_epc_nationtwo_listed_buildings_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge listed buildings datasets from two nations into a single dataset.
    Args:
        enhanced_epc_nationone_listed_buildings_df (pl.DataFrame): EPC dataset with listed buildings from nation one
        enhanced_epc_nationtwo_listed_buildings_df (pl.DataFrame): EPC dataset with listed buildings from nation two
    Returns:
        enhanced_epc_with_all_listed_buildings (pl.DataFrame): EPC dataset with listed buildings from both nations
    """
    enhanced_epc_with_all_listed_buildings = pl.concat(
        [
            enhanced_epc_nationone_listed_buildings_df,
            enhanced_epc_nationtwo_listed_buildings_df,
        ],
        how="vertical",
    )
    enhanced_epc_with_all_listed_buildings = (
        enhanced_epc_with_all_listed_buildings.group_by("UPRN").agg(
            [
                pl.col("*")
                .exclude("Grade")
                .first()
                .name.keep(),  # Keep the first occurrence of all columns except 'Grade'
                pl.col("Grade")
                .map_elements(
                    lambda x: ", ".join(set(str(i) for i in x if i is not None)),
                    return_dtype=pl.Utf8,
                )
                .alias("Grade"),  # Merge 'Grade' values into a single column
            ]
        )
    )
    return enhanced_epc_with_all_listed_buildings
