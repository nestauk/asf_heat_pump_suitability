import pandas as pd
import geopandas as gpd
import regex as re
import logging
from tqdm import tqdm
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, get_datasets
from asf_heat_pump_suitability.utils import geo_utils


def transform_gdf_council_bounds(
    ladnm_col: str,
    use_cols: list,
) -> gpd.GeoDataFrame:
    """
    Transform council (local authority district) geo dataset by standardising council names (lower and remove special
    characters and spaces). CRS: British National Grid (EPSG: 27700).

    Args:
        ladnm_col (str): name of column with council (LAD) names
        use_cols (list): names of columns to retain

    Returns:
        gpd.GeoDataFrame: dataframe of council (LAD) names with geometries in CRS British National Grid
    """
    council_bounds = get_datasets.load_gdf_ons_council_bounds()
    council_bounds = council_bounds[use_cols]
    council_bounds["council_name_std"] = _standardise_list_council_names(
        council_bounds[ladnm_col]
    )

    return council_bounds


def _standardise_list_council_names(name_series: pd.Series) -> list:
    """
    Standardise council (local authority district) names.

    Args:
        name_series (pd.Series): council names

    Returns:
        list: standardised council names
    """
    council_names = [
        # Replace spaces, full stops, numbers, other punctuation with "_"
        re.sub("[^a-zA-Z-,]+", "_", nm).lower().split(",")[0]
        for nm in name_series
    ]
    return council_names


def generate_gdf_map_file_to_bounds(
    land_extent_location: str = config["data_source"]["EW_inspire_land_extent"],
    ladnm_col: str = "LAD23NM",
    use_cols: list = ["LAD23NM", "LAD23CD", "geometry"],
    save_as: str = None,
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame with land extent (INSPIRE) files and their bounding polygons by matching land extent filenames
    to ONS council polygon names. CRS: British National Grid (EPSG: 27700).

    Args:
        land_extent_location (str): location of land extent (INSPIRE) files. Defaults to S3 location.
        ladnm_col (str): name of column with council (LAD) names in council polygons file
        use_cols (list): names of columns to keep in council polygons file. Must include the following columns:
        council/LAD name, council/LAD code, council/LAD geometry.
        save_as (str): path to save matched files to. Optional.

    Returns:
        gpd.GeoDataFrame: land extent (INSPIRE) files and their bounding polygons in British National Grid CRS
    """
    council_bounds = transform_gdf_council_bounds(ladnm_col, use_cols)
    land_extent_file_names = base_getters.list_files_s3_location(land_extent_location)

    matches = _match_list_file_to_name(
        land_extent_files=land_extent_file_names,
        council_names=council_bounds["council_name_std"],
    )

    file_to_bounds = pd.DataFrame(
        {"inspire_file_name": land_extent_file_names, "council_bounds_matches": matches}
    ).merge(
        council_bounds,
        how="left",
        left_on="council_bounds_matches",
        right_on="council_name_std",
    )

    file_to_bounds = gpd.GeoDataFrame(
        file_to_bounds, crs="EPSG:27700", geometry="geometry"
    )

    use_cols.append("council_name_std")

    file_to_bounds = fill_nulls_file_bounds(
        file_to_bounds, council_bounds, ladnm_col, use_cols
    )

    if save_as:
        file_to_bounds.to_file(save_as)

    return file_to_bounds


def _match_list_file_to_name(land_extent_files: list, council_names: pd.Series) -> list:
    """
    Match land extent (INSPIRE) file names to council names.

    Args:
        land_extent_files (list): land extent file names
        council_names (pd.Series): standardised council / local authority district names

    Returns:
        list: council name matched to land extent (INSPIRE) file name
    """
    matches = []
    for f in land_extent_files:
        match = [council for council in council_names if council in f.lower()]
        if not len(match):
            match = None
        else:
            # Select longest string match
            match = max(match, key=len)
        matches.append(match)

    return matches


def fill_nulls_file_bounds(
    gdf: gpd.GeoDataFrame,
    council_bounds: gpd.GeoDataFrame,
    ladnm_col: str,
    fill_cols: list,
) -> gpd.GeoDataFrame:
    """
    Fill missing file bounding polygons for land extent (INSPIRE) files.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame of land extent (INSPIRE) files and file polygons
        council_bounds (gpd.GeoDataFrame): GeoDataFrame of council / Local Authority District (LAD) boundaries.
        ladnm_col (str): name of column with council (LAD) names in council polygons file
        fill_cols (list): names of columns to fill with council (LAD) metadata (LAD code, name, geometry, and
        standardised name).

    Returns:
        gpd.GeoDataFrame: land extent (INSPIRE) files with file bounding polygons
    """
    missing_bbox = gdf[gdf["council_bounds_matches"].isnull()][
        "inspire_file_name"
    ].to_list()

    logging.info(
        f"Filling missing council boundaries for {len(missing_bbox)} land extent (INSPIRE) files"
    )
    for file in tqdm(missing_bbox):
        land_parcels_gdf = get_datasets.load_gdf_inspire_land_parcels(f"s3://{file}")
        # Get the council name for the majority of a sample of land polygon centres
        candidate_nm = (
            gpd.sjoin(
                land_parcels_gdf.sample(500).centroid.to_frame("geometry"),
                council_bounds,
            )[ladnm_col]
            .value_counts()
            .index[0]
        )
        # Select council / local authority district that 'contains' majority of land centroids
        candidate = council_bounds.loc[council_bounds[ladnm_col] == candidate_nm]
        # Update gdf with selected council bounds
        gdf.loc[
            gdf["inspire_file_name"] == file,
            fill_cols,
        ] = candidate.to_numpy()[0]

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
