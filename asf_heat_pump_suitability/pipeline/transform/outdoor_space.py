"""
Functions to calculate outdoor space estimates from building footprints and land extents.
"""

import logging

import geopandas as gpd
import pandas as pd
import polars as pl

# Maximum footprint area (m²) at which a building is treated as an outbuilding
# (e.g. garage, shed).  30 m² is approximately the floor area of a large double garage.
OUTBUILDING_SIZE_M2: int = 30

# For buildings whose total area <= OUTBUILDING_SIZE_M2 we keep the intersection only if
# the intersecting slice is at least this proportion of the building's area.
# 0.45 avoids retaining a tiny corner sliver while still capturing outbuildings that
# straddle a parcel boundary.
SMALL_BUILDING_MIN_OVERLAP_PROPORTION: float = 0.45

# For buildings whose total area > OUTBUILDING_SIZE_M2 we keep the intersection only if
# its area is at least this many m².  15 m² is roughly the footprint of a small shed and
# is large enough to rule out coordinate-tolerance slivers at parcel boundaries.
MIN_INTERSECTION_AREA_M2: float = 15


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
    file_matches = pd.Series(gdf["ms_url"].values, index=gdf["inspire_file_name"]).sort_index()

    return file_matches


def generate_gdf_building_intersections(
    land_parcels_gdf: gpd.GeoDataFrame,
    building_footprints_gdf: gpd.GeoDataFrame,
    outbuilding_size: int = OUTBUILDING_SIZE_M2,
    s_building_prop: float = SMALL_BUILDING_MIN_OVERLAP_PROPORTION,
    min_intersection: float = MIN_INTERSECTION_AREA_M2,
) -> gpd.GeoDataFrame:
    """
    Generate intersections between land parcel polygons and building footprint polygons, with rules applied to remove
    suspected erroneous building intersections (i.e. small corners or slivers of buildings) but retain outbuildings.
    Building intersections are kept if they meet one of the following conditions:
    1. total building area <= outbuilding_size and building intersection area >= s_building_prop * outbuilding_size
    2. total building area > outbuilding_size and building intersection area >= min_intersection

    Together these conditions aim:
    - handle buildings which are equal to or smaller than the outbuilding size to retain true small
    segments of buildings that could represent garages, sheds, or other outbuildings.
    - handle buildings that are larger than the outbuilding size, keeping building intersections that
    are large enough to represent a meaningful portion of a main building
    - remove intersections that likely represent small corners or boundary-touching errors

    Args:
        land_parcels_gdf (gpd.GeoDataFrame): land parcel polygons
        building_footprints_gdf (gpd.GeoDataFrame): building footprint polygons
        outbuilding_size (int): max area (m2) of building footprints assumed to be outbuildings. Default 30m2 - the average size of a double garage.
        s_building_prop (float): minimum proportion of buildings smaller than outbuilding_size. Default 45%.
        min_intersection (float): minimum area (m2) of intersection of buildings larger than outbuilding_size (metres squared). Default 15m2 - the minimum size for a building intersection to be considered a genuine building.

    Returns:
        gpd.GeoDataFrame: polygon intersections of land parcel polygons and building footprint polygons
    """
    building_footprints_gdf["building_area_m2"] = building_footprints_gdf.area

    gdf = gpd.overlay(
        land_parcels_gdf,
        building_footprints_gdf,
        how="intersection",
        keep_geom_type=False,
    )
    gdf["building_intersection_area_m2"] = gdf["geometry"].area
    gdf["min_size_of_small_building"] = gdf["building_area_m2"] * s_building_prop

    gdf = gdf[
        (
            (gdf["building_area_m2"] <= outbuilding_size)
            & (gdf["building_intersection_area_m2"] >= gdf["min_size_of_small_building"])
        )
        | ((gdf["building_area_m2"] > outbuilding_size) & (gdf["building_intersection_area_m2"] >= min_intersection))
    ]

    # Drop intersecting geometries which are Points or LineStrings
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"]

    return gdf


def generate_gdf_outdoor_space(
    building_intersections_gdf: gpd.GeoDataFrame, land_parcels_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame of land extents with total and max contiguous outdoor space areas.

    Args:
        building_intersections_gdf (gpd.GeoDataFrame): intersections of land parcel polygons and building footprint polygons
        land_parcels_gdf (gpd.GeoDataFrame): land parcel polygons

    Returns:
        gpd.GeoDataFrame: original land parcel polygons with calculated total and max contiguous outdoor space area (m2)
    """
    # Get land minus buildings - MultiPolygons will be created for land parcels which get split into multi-parts
    land_minus_buildings = gpd.overlay(
        land_parcels_gdf,
        building_intersections_gdf,
        how="difference",
        keep_geom_type=False,
    )

    # Explode multi-part geometries into single part geometries to get largest piece
    land_minus_buildings_parts = land_minus_buildings.explode(index_parts=False)
    land_minus_buildings_parts["outdoor_space_area_m2"] = land_minus_buildings_parts.geometry.area

    # Keep max size and total outdoor area
    outdoor_space_df = (
        # Group land extent intersections by their ID to get the largest part per parcel
        land_minus_buildings_parts.groupby("NATIONALCADASTRALREFERENCE")
        .agg(
            max_contiguous_outdoor_space_area_m2=("outdoor_space_area_m2", max),
            total_outdoor_space_area_m2=("outdoor_space_area_m2", sum),
        )
        .reset_index(
            # Drop duplicates in case there are multiple max values
        )
        .drop_duplicates(subset="NATIONALCADASTRALREFERENCE")
    )

    # Merge full land extent geometry back on
    outdoor_space_gdf = gpd.GeoDataFrame(
        outdoor_space_df.merge(
            land_parcels_gdf[["NATIONALCADASTRALREFERENCE", "geometry"]],
            how="left",
            on="NATIONALCADASTRALREFERENCE",
        )
    )

    return outdoor_space_gdf


def deduplicate_df_outdoor_space(df: pl.DataFrame) -> pl.DataFrame:
    """
    Deduplicate UPRNs matched to multiple land extents by keeping the one with the smallest total outdoor space area.

    Args:
        df (pl.DataFrame): UPRNs with outdoor space estimates

    Returns:
        pl.DataFrame: deduplicated UPRNs with outdoor space estimates
    """
    df = df.with_columns(pl.col("UPRN").is_duplicated().alias("UPRN_duplicated"))

    _deduplicated_df = (
        df.filter(pl.col("UPRN_duplicated"))
        .with_columns(min_total=pl.col("total_outdoor_space_area_m2").min().over("UPRN"))
        .filter(
            # Get smallest outdoor space
            pl.col("total_outdoor_space_area_m2") == pl.col("min_total")
            # Deduplicate - any duplicates will now (most likely) have the same total outdoor space
        )
        .unique(subset="UPRN")
        .drop("min_total")
    )

    return pl.concat([df.filter(~pl.col("UPRN_duplicated")), _deduplicated_df]).drop("UPRN_duplicated")


def sjoin_df_uprn_to_outdoor_space(uprns_gdf: gpd.GeoDataFrame, outdoor_space_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Join outdoor space estimates to UPRNs. UPRNs will be assigned the outdoor space calculated for the land extent parcel
    that they are contained within.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs and their point geometries
        outdoor_space_gdf (gpd.GeoDataFrame): original land parcel polygons with calculated total and max contiguous outdoor space area (m2)

    Returns:
        pl.DataFrame: UPRNs with estimated outdoor space (m2)
    """
    return pl.from_pandas(
        uprns_gdf.sjoin(outdoor_space_gdf, how="left", predicate="within").drop(columns=["geometry", "index_right"])
    )
