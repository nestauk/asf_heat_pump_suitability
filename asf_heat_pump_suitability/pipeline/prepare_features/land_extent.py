import pandas as pd
import geopandas as gpd
import shapely
import regex as re
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, get_datasets
from asf_heat_pump_suitability.utils import geo_utils


def transform_gdf_council_bounds(
    ladnm_col: str,
    keep_cols: list,
) -> gpd.GeoDataFrame:
    """
    Transform council (local authority district) geo dataset: standardise council names (lower and remove special
    characters and spaces) and convert geometries to British National Grid CRS.

    Args:
        ladnm_col (str): name of column with council (LAD) names
        keep_cols (list): names of columns to keep

    Returns:
        gpd.GeoDataFrame: dataframe of council (LAD) names with geometries in British National Grid CRS
    """
    council_bounds = get_datasets.load_gdf_ons_council_bounds()
    council_bounds = council_bounds[keep_cols]
    council_bounds["council_name_std"] = _standardise_list_council_names(
        council_bounds[ladnm_col]
    )
    council_bounds = council_bounds.to_crs(epsg="27700")

    return council_bounds


def generate_gdf_map_file_to_bounds(
    land_extent_location: str = config["data_source"]["EW_inspire_land_extent"],
    ladnm_col: str = "LAD23NM",
    keep_cols: list = ["LAD23NM", "LAD23CD", "geometry"],
) -> gpd.GeoDataFrame:
    """
    Create dataframe with land extent filenames and their matching bounding polygons by matching land extent filenames
    to ONS council polygon names.

    Args:
        land_extent_location (str): location of land extent files. Default S3 location.
        ladnm_col (str): name of column with council (LAD) names in council polygons file
        keep_cols (list): names of columns to keep in council polygons file

    Returns:
        gpd.GeoDataFrame: dataframe with land extent filenames and their matching bounding polygons
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

    return file_to_bounds


def _standardise_list_council_names(name_series):
    """ """
    lad_names = [
        re.sub("[^a-zA-Z-,]+", "_", ln).lower().split(",")[0] for ln in name_series
    ]
    return lad_names


def _match_list_file_to_name(land_extent_files, council_names):
    """ """
    matches = []
    for f in land_extent_files:
        f_matches = [lad for lad in council_names if lad in f.lower()]
        if not len(f_matches):
            f_matches = None
        matches.append(f_matches)

    return matches


def fill_nulls_file_bounds(gdf):
    """ """
    missing_bbox = gdf[gdf["council_bounds_matches"].isnull()][
        "inspire_file_name"
    ].to_list()
    for file in missing_bbox:
        file_polygon = get_polygon_file_bounds(f"s3://{file}")
        gdf.loc[gdf["inspire_file_name"] == file, "geometry"] = file_polygon

    return gdf


def get_polygon_file_bounds(path: str) -> shapely.Polygon:
    """ """
    gdf = get_datasets.load_gdf_inspire_land_parcels(path)
    file_bounds = gdf["geometry"].total_bounds

    bounds = {
        "minx": file_bounds[0],
        "miny": file_bounds[1],
        "maxx": file_bounds[2],
        "maxy": file_bounds[3],
    }

    bbox_polygon = shapely.Polygon(
        [  # TODO: needs converting to BNG
            [bounds["minx"], bounds["miny"]],
            [bounds["minx"], bounds["maxy"]],
            [bounds["maxx"], bounds["maxy"]],
            [bounds["maxx"], bounds["miny"]],
            [bounds["minx"], bounds["miny"]],
        ]
    )

    return bbox_polygon


def transform_gdf_land_parcels(inspire_file):
    """ """
    gdf = get_datasets.load_gdf_inspire_land_parcels(inspire_file)
    gdf = gdf[["NATIONALCADASTRALREFERENCE", "geometry"]]
    gdf = geo_utils.transform_gdf_drop_close_duplicates(gdf)
    gdf["land_area"] = gdf["geometry"].area

    return gdf
