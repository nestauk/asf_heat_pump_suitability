import pandas as pd
import geopandas as gpd
import regex as re
from typing import List
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, get_datasets
from asf_heat_pump_suitability.utils import geo_utils


def transform_gdf_council_bounds(
    ladnm_col: str,
    keep_cols: list,
) -> gpd.GeoDataFrame:
    """
    Transform council (local authority district) geo dataset: standardise council names (lower and remove special
    characters and spaces), CRS: British National Grid (EPSG: 27700).

    Args:
        ladnm_col (str): name of column with council (LAD) names.
        keep_cols (list): names of columns to keep

    Returns:
        gpd.GeoDataFrame: dataframe of council (LAD) names with geometries in British National Grid CRS
    """
    council_bounds = get_datasets.load_gdf_ons_council_bounds()
    council_bounds = council_bounds[keep_cols]
    council_bounds["council_name_std"] = _standardise_list_council_names(
        council_bounds[ladnm_col]
    )

    return council_bounds


def generate_gdf_map_file_to_bounds(
    land_extent_location: str = config["data_source"]["EW_inspire_land_extent"],
    ladnm_col: str = "LAD23NM",
    keep_cols: list = ["LAD23NM", "LAD23CD", "geometry"],
    save_as: str = None,
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame with land extent files and their bounding polygons by matching land extent filenames
    to ONS council polygon names.

    Args:
        land_extent_location (str): location of land extent files. Default S3 location.
        ladnm_col (str): name of column with council (LAD) names in council polygons file
        keep_cols (list): names of columns to keep in council polygons file
        save_as (str): path to save matched files to. Optional.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with land extent files and their bounding polygons
    """
    council_bounds = transform_gdf_council_bounds(ladnm_col, keep_cols)
    land_extent_files = base_getters.list_files_s3_location(land_extent_location)

    matches = _match_list_file_to_name(
        land_extent_files=land_extent_files,
        council_names=council_bounds["council_name_std"],
    )

    matches = pd.DataFrame(
        {"inspire_file_name": land_extent_files, "council_bounds_matches": matches}
    ).explode("council_bounds_matches")

    file_to_bounds = matches.merge(
        council_bounds,
        how="left",
        left_on="council_bounds_matches",
        right_on="council_name_std",
    )

    file_to_bounds = gpd.GeoDataFrame(
        file_to_bounds, crs="EPSG:27700", geometry="geometry"
    )
    file_to_bounds = fill_nulls_file_bounds(file_to_bounds)
    if save_as:
        file_to_bounds.to_file(save_as, crs="EPSG:27700")

    return file_to_bounds


def _standardise_list_council_names(name_series: pd.Series) -> list:
    """
    Standardise council names.

    Args:
        name_series (pd.Series): council names

    Returns:
        list: standardised council names
    """
    lad_names = [
        # Replace spaces, full stops, numbers, other punctuation with "_"
        re.sub("[^a-zA-Z-,]+", "_", ln).lower().split(",")[0]
        for ln in name_series
    ]
    return lad_names


def _match_list_file_to_name(
    land_extent_files: list, council_names: pd.Series
) -> List[list]:
    """
    Match land extent files to council names. Some files will match to multiple council names e.g. the file for "Wyre"
    will match with "Wyre" and "Wyre Forest" councils.

    Args:
        land_extent_files (list):
        council_names (pd.Series):

    Returns:
        List[list]: lists of council names matched to file name
    """
    matches = []
    for f in land_extent_files:
        f_matches = [lad for lad in council_names if lad in f.lower()]
        if not len(f_matches):
            f_matches = None
        matches.append(f_matches)

    return matches


def fill_nulls_file_bounds(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fill file bounds polygons for INSPIRE land extent files with no file polygons.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame of INSPIRE land extent files and file polygons

    Returns:
        gpd.GeoDataFrame: land extent files with file polygons
    """
    missing_bbox = gdf[gdf["council_bounds_matches"].isnull()][
        "inspire_file_name"
    ].to_list()

    for file in missing_bbox:
        land_parcels_gdf = get_datasets.load_gdf_inspire_land_parcels(f"s3://{file}")
        file_polygon = geo_utils.get_polygon_gdf_bounds(land_parcels_gdf)
        gdf.loc[gdf["inspire_file_name"] == file, "geometry"] = file_polygon

    return gdf


def transform_gdf_land_parcels(land_parcel_file: str) -> gpd.GeoDataFrame:
    """
    Load and transform land parcel file to produce GeoDataFrame with unique National Cadastral Reference, land extent
    geometry, and land area (m2).

    Args:
        land_parcel_file (str): name of land parcel file

    Returns:
        gpd.GeoDataFrame: land parcel geodata
    """
    gdf = get_datasets.load_gdf_inspire_land_parcels(land_parcel_file)
    gdf = gdf[["NATIONALCADASTRALREFERENCE", "geometry"]]
    gdf = geo_utils.transform_gdf_drop_duplicates(gdf)
    gdf["land_area_m2"] = gdf["geometry"].area

    return gdf
