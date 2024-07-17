import geopandas as gpd
import logging
import pandas as pd
from collections.abc import Mapping


def match_dict_files_land_building(
    land_files_gdf: gpd.GeoDataFrame, building_files_gdf: gpd.GeoDataFrame
) -> Mapping:
    """
    Get dictionary of intersecting INSPIRE land parcel files to Microsoft building footprint files.

    Args:
        land_files_gdf (gpd.GeoDataFrame): INSPIRE land file names and file bounding polygons
        building_files_gdf (gpd.GeoDataFrame): Microsoft building footprint file names and file bounding polygons

    Returns:
        Mapping: mapping where keys are INSPIRE land parcel file names and values are Microsoft building footprint file
        names
    """
    logging.info("Mapping building footprint files to INSPIRE land registry files")
    gdf = gpd.overlay(land_files_gdf, building_files_gdf, how="intersection")
    gdf = gdf.sort_values(by="inspire_file_name")
    file_dict = pd.Series(
        gdf["ms_url"].values, index=gdf["inspire_file_name"]
    ).to_dict()

    return file_dict


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
    gdf = gpd.overlay(land_parcels_gdf, building_footprints_gdf, how="intersection")
    gdf["building_intersection_area_m2"] = gdf["geometry"].area

    gdf["min_size_of_small_building"] = gdf["building_area_m2"] * s_building_prop
    gdf["min_size_of_large_building"] = gdf["building_area_m2"] * l_building_prop

    gdf = gdf[
        (
            (gdf["building_intersection_area_m2"] <= outbuilding_size)
            & (
                gdf["building_intersection_area_m2"]
                >= gdf["min_size_of_small_building"]
            )
        )
        | (
            (gdf["building_intersection_area_m2"] > outbuilding_size)
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
        intersections (gpd.GeoDataFrame): intersections of land parcel polygons and building footprint polygons
        land_parcels (gpd.GeoDataFrame): land parcel polygons

    Returns:
        gpd.GeoDataFrame: land parcels matched to building sections with total garden area (m2) calculated
    """
    building_size = (
        intersections_gdf.groupby("NATIONALCADASTRALREFERENCE")[
            ["building_id", "building_intersection_area_m2", "height"]
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
            max_building_height=(
                "height",
                "max",
            ),  # get max building height on land parcel
        )
        .reset_index()
    )

    gardens_gdf = land_parcels_gdf.merge(
        building_size, how="left", on="NATIONALCADASTRALREFERENCE"
    )

    gardens_gdf["garden_area_m2"] = (
        gardens_gdf["land_area_m2"] - gardens_gdf["total_building_area_m2"]
    )

    return gardens_gdf
