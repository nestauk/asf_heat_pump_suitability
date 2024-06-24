import geopandas as gpd
import logging
import pandas as pd


def match_dict_files_inspire_microsoft(land_files_gdf, microsoft_files_gdf):
    """
    Get dict of matching files. Land parcel files to Microsoft building footprint files
    """
    logging.info("Mapping building footprint files to INSPIRE land registry files")
    gdf = gpd.overlay(land_files_gdf, microsoft_files_gdf, how="intersection")
    gdf = gdf.sort_values(by="inspire_file_name")
    file_dict = pd.Series(
        gdf["ms_url"].values, index=gdf["inspire_file_name"]
    ).to_dict()

    return file_dict


def generate_gdf_land_building_overlay(
    land_parcels,
    building_footprints,
    outbuilding_size: int = 30,
    s_building_prop: float = 0.45,
    l_building_prop: float = 0.05,
):
    """
    Intersection
    """
    gdf = gpd.overlay(land_parcels, building_footprints, how="intersection")
    gdf["building_intersection_area"] = gdf["geometry"].area

    gdf["min_of_building_small"] = gdf["building_area"] * s_building_prop
    gdf["min_of_building_large"] = gdf["building_area"] * l_building_prop

    gdf = gdf[
        (
            (gdf["building_intersection_area"] <= outbuilding_size)
            & (gdf["building_intersection_area"] >= gdf["min_of_building_small"])
        )
        | (
            (gdf["building_intersection_area"] > outbuilding_size)
            & (gdf["building_intersection_area"] >= gdf["min_of_building_large"])
        )
    ]

    return gdf


def generate_gdf_garden_size(intersection, land_parcels):
    """
    Args:
        gdf: overlay intersection (output of function above)
    """
    building_size = (
        intersection.groupby("NATIONALCADASTRALREFERENCE")[
            ["index", "building_intersection_area", "height"]
        ]
        .agg(
            building_ids=("building_id", list),
            total_building_area_m2=("building_intersection_area", "sum"),
            max_building_height=("height", "max"),
        )
        .reset_index()
    )

    gardens = land_parcels.merge(
        building_size, how="left", on="NATIONALCADASTRALREFERENCE"
    )

    return gardens
