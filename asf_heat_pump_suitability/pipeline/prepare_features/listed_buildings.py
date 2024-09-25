import polars as pl
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import lat_lon


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


def generate_df_epc_listed_buildings(
    epc_df: pl.DataFrame, nations: list = ["England", "Scotland", "Wales"]
):
    """
    Generate dataframe of listed buildings in EPC data in specified nation(s).

    Args:
        epc_df (pl.DataFrame):
        nations (list):

    Returns:
        pl.DataFrame:
    """
    dfs = []
    for nation in nations:
        logging.info(f"Loading listed building data for {nation}")
        gdf = transform_gdf_listed_buildings(nation)
        df = sjoin_df_epc_with_nation_listed_buildings(epc_df, gdf)
        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def sjoin_df_epc_with_nation_listed_buildings(
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
        epc_gdf = lat_lon.generate_gdf_uprn_coords(df=epc_chunk)[["UPRN", "geometry"]]
        df = epc_gdf.sjoin(listed_buildings_gdf, how="inner", predicate="intersects")[
            ["UPRN", "listed_building_grade"]
        ].drop_duplicates(subset="UPRN")

        dfs.append(df)

    df = pl.from_pandas(pd.concat(dfs))
    df = df.with_columns(
        pl.when(pl.col("listed_building_grade").is_null())
        .then(False)
        .otherwise(True)
        .alias("listed_building"),
    )
    return df.select(["PUPRN", "listed_building"])
