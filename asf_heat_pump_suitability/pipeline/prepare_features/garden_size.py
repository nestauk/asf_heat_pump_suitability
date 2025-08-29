import geopandas as gpd
import logging
import pandas as pd
import polars as pl


def match_series_files_land_building(
    land_files_gdf: gpd.GeoDataFrame, building_files_gdf: gpd.GeoDataFrame
) -> pd.Series:
    """
    Get Series of intersecting INSPIRE land parcel files and Microsoft building footprint files.

    Args:
        land_files_gdf (gpd.GeoDataFrame): INSPIRE land file names and file bounding polygons
        building_files_gdf (gpd.GeoDataFrame): Microsoft building footprint file names and file bounding polygons

    Returns:
        pd.Series: Series where indices are INSPIRE land parcel file names and values are Microsoft building footprint files
        names
    """
    logging.info("Mapping building footprint files to INSPIRE land registry files")
    gdf = land_files_gdf.sjoin(building_files_gdf, how="inner", predicate="intersects")
    file_matches = pd.Series(
        gdf["ms_url"].values, index=gdf["inspire_file_name"]
    ).sort_index()

    return file_matches


def generate_gdf_land_building_overlay(
    land_parcels_gdf: gpd.GeoDataFrame,
    building_footprints_gdf: gpd.GeoDataFrame,
    outbuilding_size: int = 30,
    s_building_prop: float = 0.45,
    l_building_prop: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Generate intersections between land parcel polygons and building footprint polygons, with rules applied to remove
    suspected erroneous building intersections (i.e. small corners or slivers of buildings) but retain outbuildings.
    Building intersections are kept if they meet the following conditions:
    1. building intersection area <= outbuilding_size and >= s_building_prop * outbuilding_size
    2. building intersection area > outbuilding_size and >= l_building_prop * outbuilding_size

    Args:
        land_parcels_gdf (gpd.GeoDataFrame): land parcel polygons
        building_footprints_gdf (gpd.GeoDataFrame): building footprint polygons
        outbuilding_size (int): max area (m2) of building footprints assumed to be outbuildings. Default 30m2.
        s_building_prop (float): minimum proportion of buildings smaller than outbuilding_size. Default 45%.
        l_building_prop (float): minimum proportion of buildings larger than outbuilding_size. Default 5%.

    Returns:
        gpd.GeoDataFrame: intersections of land parcel polygons and building footprint polygons
    """
    gdf = gpd.overlay(
        land_parcels_gdf,
        building_footprints_gdf,
        how="intersection",
        keep_geom_type=False,
    )
    gdf["building_intersection_area_m2"] = gdf["geometry"].area

    gdf["min_size_of_small_building"] = gdf["building_area_m2"] * s_building_prop
    gdf["min_size_of_large_building"] = gdf["building_area_m2"] * l_building_prop

    gdf = gdf[
        (
            (gdf["building_area_m2"] <= outbuilding_size)
            & (
                gdf["building_intersection_area_m2"]
                >= gdf["min_size_of_small_building"]
            )
        )
        | (
            (gdf["building_area_m2"] > outbuilding_size)
            & (
                gdf["building_intersection_area_m2"]
                >= gdf["min_size_of_large_building"]
            )
        )
    ]

    return gdf


def generate_gdf_garden_size(
    intersections_gdf: gpd.GeoDataFrame, land_parcels_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame of land parcels matched to building sections with data on land, building, and garden area,
    and maximum building height (where available).

    Args:
        intersections_gdf (gpd.GeoDataFrame): intersections of land parcel polygons and building footprint polygons
        land_parcels_gdf (gpd.GeoDataFrame): land parcel polygons

    Returns:
        gpd.GeoDataFrame: land parcels matched to building sections with total garden area (m2) calculated with geometry
        of land parcels
    """
    building_size = (
        intersections_gdf.groupby("NATIONALCADASTRALREFERENCE")[
            [
                "building_id",
                "building_intersection_area_m2",
                # "height",
                "source",
            ]
        ]
        .agg(
            building_ids=(
                "building_id",
                list,
            ),  # get list of building IDs matched to one land parcel
            total_building_area_m2=(
                "building_intersection_area_m2",
                "sum",
            ),  # get total building area on land parcel
            # max_building_height=(
            #     "height",
            #     "max",
            # ),  # get max building height on land parcel
        )
        .reset_index()
    )

    gardens_gdf = land_parcels_gdf.merge(
        building_size, how="inner", on="NATIONALCADASTRALREFERENCE"
    )

    gardens_gdf["garden_area_m2"] = (
        gardens_gdf["land_area_m2"] - gardens_gdf["total_building_area_m2"]
    )

    return gardens_gdf


def deduplicate_df_garden_size(df: pl.DataFrame) -> pl.DataFrame:
    """
    Deduplicate UPRNs matched to multiple gardens by taking the average size of the multiple gardens (for gardens
    below a threshold size).

    Args:
        df (pl.DataFrame): UPRNs with garden size estimates

    Returns:
        pl.DataFrame: deduplicated UPRNs with garden size estimates
    """
    df = df.with_columns(pl.col("UPRN").is_duplicated().alias("UPRN_duplicated"))
    # Remove gardens with area above the 97th percentile if they are matched to duplicate UPRNs
    df = df.filter(
        ~(
            pl.col("UPRN_duplicated")
            & (pl.col("garden_area_m2") > df["garden_area_m2"].quantile(quantile=0.97))
        )
    )
    # Calculate median garden size for UPRNs with multiple gardens
    df = df.group_by("UPRN").agg(pl.median("garden_area_m2"))

    return df
