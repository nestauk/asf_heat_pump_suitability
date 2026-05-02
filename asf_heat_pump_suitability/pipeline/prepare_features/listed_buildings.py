import polars as pl
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import lat_lon


def generate_df_epc_listed_buildings(
    epc_df: pl.DataFrame, nations: list = ["England", "Scotland", "Wales"]
) -> pl.DataFrame:
    """
    Generate dataframe of listed buildings in EPC data in specified nation(s).

    Args:
        epc_df (pl.DataFrame): EPC dataset with x, y coordinates per UPRN
        nations (list): nation(s) to load listed buildings data for. Options: "England"; "Scotland"; "Wales".

    Returns:
        pl.DataFrame: EPC UPRNs in listed buildings
    """
    dfs = []
    for nation in nations:
        logging.info(f"Loading listed buildings data for {nation}")
        gdf = transform_gdf_listed_buildings(nation)
        dfs.append(chunk_sjoin_df_epc_listed_buildings(epc_df, gdf))

    return pl.concat(dfs, how="vertical")


def transform_gdf_listed_buildings(nation: str = "GB") -> gpd.GeoDataFrame:
    """
    Load and transform listed buildings polygons dataset for specified nation.

    Args:
        nation (str): nation to load listed buildings data for. Options: "England"; "Scotland"; "Wales"; "GB". Defaults to "GB" which loads data for all Great Britain.

    Returns:
        gpd.GeoDataFrame: listed buildings dataset for specified nation with grade and geometry columns.
    """
    gdf = get_datasets.load_gdf_listed_buildings(nation, columns=["geometry"])
    gdf = gdf.drop_duplicates(subset="geometry")
    gdf["in_listed_building"] = True

    return gdf


def extend_df_listed_building_bool(
    features_df: pl.DataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    listed_buildings_gdf: gpd.GeoDataFrame,
) -> pl.DataFrame:
    """
    Add boolean column to features_df indicating whether UPRN is in a listed building.

    Args:
        features_df (pl.DataFrame): dataframe with one row per UPRN and UPRN column.
        uprns_gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries for each UPRN.
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame of building footprints with geometry column.
        listed_buildings_gdf (gpd.GeoDataFrame): GeoDataFrame of listed buildings with geometry column.
            Can be point or polygon geometries.

    Returns:
        pl.DataFrame: input features_df with new boolean column `in_listed_building` indicating whether UPRN is in a listed building.
    """

    # Separate geoemtries into points and polygons and spatial join separately to account

    # Identify listed buildings that have polygon geoemetries
    listed_polygons_gdf = listed_buildings_gdf[
        listed_buildings_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ][["geometry", "in_listed_building"]]

    # Identify listed buildings that have point geometries
    listed_points_gdf = listed_buildings_gdf[
        listed_buildings_gdf.geometry.type.isin(["Point", "MultiPoint"])
    ][["geometry", "in_listed_building"]]

    # TODO: we need to double check that this isn't labelling buildings which touch a listed building on one edge as 'listed'.
    # Join for polygons: We want buildings that touch/overlap listed building polygons
    joined_polys_gdf = buildings_gdf.sjoin(
        listed_polygons_gdf, how="inner", predicate="intersects"
    )

    # Join for points: We want buildings that contain the listed building points
    joined_pts_gdf = buildings_gdf.sjoin(
        listed_points_gdf, how="inner", predicate="contains"
    )

    # Combine joined polygons and points
    joined_gdf = pd.concat([joined_polys_gdf, joined_pts_gdf], ignore_index=True)

    # Spatial join UPRNs with listed buildings to label UPRNs in listed buildings.
    uprns_gdf = (
        uprns_gdf.sjoin(
            joined_gdf[["geometry", "in_listed_building"]],
            how="left",
            predicate="intersects",
        )
        .fillna({"in_listed_building": False})
        .drop(columns="index_right")
    )

    # Extend features_df with listed building flag
    features_df = features_df.join(
        pl.from_pandas(uprns_gdf[["UPRN", "in_listed_building"]]),
        how="left",
        on="UPRN",
    )

    return features_df


def chunk_sjoin_df_epc_listed_buildings(
    epc_df: pl.DataFrame,
    listed_buildings_gdf: gpd.GeoDataFrame,
    chunk_size: int = 100000,
) -> pl.DataFrame:
    """
    Chunk EPC data and spatial join EPC UPRNs with listed buildings.

    Args:
        epc_df (pl.DataFrame): Enhanced EPC dataset with X and Y coordinates
        listed_buildings_gdf (gpd.GeoDataFrame): listed buildings GeoDataFrame. Can be points / polygons.
        chunk_size (int): number of EPC rows in each partition. Default 100,000

    Returns:
        pl.DataFrame: EPC UPRNs in listed buildings
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
        df = sjoin_df_epc_listed_buildings(epc_chunk, listed_buildings_gdf)
        dfs.append(df)

    df = pl.from_pandas(pd.concat(dfs))

    return df.select(["UPRN", "listed_building"])


def sjoin_df_epc_listed_buildings(
    epc_df: pl.DataFrame, listed_buildings_gdf: gpd.GeoDataFrame, distance: float = 10
) -> pd.DataFrame:
    """
    Spatial join EPC UPRNs with listed buildings using `geopandas.GeoDataFrame.sjoin_nearest` where Point or MultiPoint
    geometries detected, and `geopandas.GeoDataFrame.sjoin` where Polygons or MultiPolygons detected.

    Args:
        epc_df (pl.DataFrame): EPC dataset with X and Y coordinates per UPRN
        listed_buildings_gdf (gpd.GeoDataFrame): listed buildings data
        distance (float): maximum distance (m) within which to query for nearest geometry where `sjoin_nearest` used.
                          Must be greater than 0. Default 10m.

    Returns:
        pd.DataFrame: EPC UPRNs in listed buildings
    """
    epc_gdf = lat_lon.generate_gdf_uprn_coords(df=epc_df, usecols=["UPRN"])

    # Some EPC rows have invalid geometry e.g. `POINT(NaN NaN)` because they have invalid UPRNs
    valid_rows = epc_gdf["geometry"].is_valid.sum()
    logging.warning(
        f"{len(epc_gdf) - valid_rows} EPC rows with invalid point geometries cannot be matched to listed buildings."
    )
    epc_gdf = epc_gdf[epc_gdf["geometry"].is_valid]

    if any(
        [
            expr in listed_buildings_gdf.geom_type.unique()
            for expr in ["Point", "MultiPoint"]
        ]
    ):
        df = epc_gdf.sjoin_nearest(
            listed_buildings_gdf, how="inner", max_distance=distance
        )[["UPRN", "listed_building"]].drop_duplicates(subset="UPRN")
    elif any(
        [
            expr in listed_buildings_gdf.geom_type.unique()
            for expr in ["Polygon", "MultiPolygon"]
        ]
    ):
        df = epc_gdf.sjoin(listed_buildings_gdf, how="inner", predicate="intersects")[
            ["UPRN", "listed_building"]
        ].drop_duplicates(subset="UPRN")
    else:
        raise ValueError(
            f"Listed buildings GeoDataFrame does not have appropriate geometries for sjoin. "
            f"Geometries required: [Multi]Point or [Multi]Polygon. "
            f"Geometries found: {listed_buildings_gdf.geom_type.unique()}"
        )
    return df
