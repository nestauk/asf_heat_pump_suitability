import pandas as pd
import geopandas as gpd
import shapely
import regex as re
from asf_heat_pump_suitability.getters import base_getters, get_datasets


def generate_gdf_map_file_to_bounds(
    land_extent_location,
    ladnm_col: str = "LAD23NM",
    keep_cols: list = ["LAD23NM", "LAD23CD", "geometry"],
):
    """ """
    council_bounds = get_datasets.load_gdf_ons_council_bounds()
    council_names = _standardise_list_council_names(council_bounds[ladnm_col])
    council_bounds = council_bounds[keep_cols]
    council_bounds["council_name_std"] = council_names

    land_extent_files = base_getters.list_files_s3_location(land_extent_location)
    matches = _match_list_file_to_name(
        land_extent_files=land_extent_files, council_names=council_names
    )

    matches = pd.DataFrame(
        {"file_name": land_extent_files, "council_bounds_matches": matches}
    )

    matches = matches.explode("council_bounds_matches")

    file_to_bounds = matches.merge(
        council_bounds,
        how="left",
        left_on="council_bounds_matches",
        right_on="council_name_std",
    )

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


def extend_gdf_parcel_bboxes(path: str) -> gpd.GeoDataFrame:
    """ """
    gdf = base_getters.load_gdf_from_s3_geojson(s3_uri=path)
    gdf["geom_types"] = gdf["geometry"].geom_type
    gdf["bbox"] = gdf["geometry"].apply(lambda x: x.bounds)
    gdf[["minx", "miny", "maxx", "maxy"]] = gpd.GeoDataFrame(
        gdf["bbox"].tolist(), index=gdf.index
    )

    return gdf


def create_dict_file_bounds(gdf) -> dict:
    """ """
    bounds = {
        "minx": gdf["minx"].min(),
        "miny": gdf["miny"].min(),
        "maxx": gdf["maxx"].max(),
        "maxy": gdf["maxy"].max(),
    }
    bounds["bbox"] = (bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])
    bounds["polygon"] = [
        [bounds["minx"], bounds["miny"]],
        [bounds["minx"], bounds["maxy"]],
        [bounds["maxx"], bounds["maxy"]],
        [bounds["maxx"], bounds["miny"]],
        [bounds["minx"], bounds["miny"]],
    ]

    return bounds
