import pandas as pd
import geopandas as gpd
import regex as re
import logging
from tqdm import tqdm
import warnings
from asf_heat_pump_suitability.getters import base_getters, load_data, load_geodata
from asf_heat_pump_suitability.utils import geo_utils


def generate_gdf_file_bounds_s(path: str) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame with land extent (INSPIRE) filenames for Scotland and their bounding polygons.
    CRS: British National Grid (EPSG: 27700).

    Args:
        path (str): S3 location of land extent (INSPIRE) files.

    Returns:
        gpd.GeoDataFrame: land extent (INSPIRE) files and their bounding polygons in British National Grid CRS
    """
    shp_dirs = base_getters.list_obj_s3_location(path)
    file_bounds = {"inspire_file_name": [], "registration_county": [], "geometry": []}
    for shp_dir in shp_dirs:
        files = base_getters.list_obj_s3_location(f"s3://{shp_dir}")
        shapefile = [file for file in files if file.endswith(".shp")][0]
        gdf = load_geodata.load_gdf_inspire_land_parcels(
            f"s3://{shapefile}", columns=["geometry"]
        )
        bounding_polygon = geo_utils.get_polygon_gdf_bounds(gdf)
        file_bounds["inspire_file_name"].append(shapefile)
        file_bounds["registration_county"].append(shp_dir.split("/")[-1])
        file_bounds["geometry"].append(bounding_polygon)

    gdf = gpd.GeoDataFrame(file_bounds, crs="EPSG:27700", geometry="geometry")

    return gdf


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
    council_bounds = load_data.load_gdf_ons_council_bounds()
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

    manual_rename = {
        "county_durham": "durham_county",
        "king_s_lynn_and_west_norfolk": "kings_lynn_and_west_norfolk",
        "kingston_upon_hull": "hull_city",
        "newcastle_upon_tyne": "newcastle_city",
    }

    council_names = [manual_rename.get(c, c) for c in council_names]

    return council_names


def generate_gdf_file_bounds_ew(
    path: str,
    ladnm_col: str = "LAD23NM",
    use_cols: list = ["LAD23NM", "LAD23CD", "geometry"],
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame with land extent (INSPIRE) filenames for England and Wales and their bounding polygons.
    CRS: British National Grid (EPSG: 27700).

    Args:
        path (str): S3 location of land extent (INSPIRE) files.
        ladnm_col (str): name of column with council (LAD) names in council polygons file
        use_cols (list): names of columns to keep in council polygons file. Must include the following columns:
        council/LAD name, council/LAD code, council/LAD geometry.

    Returns:
        gpd.GeoDataFrame: land extent (INSPIRE) files and their bounding polygons in British National Grid CRS

    Raises:
        AssertionError: if more than 10% of INSPIRE .gml files are missing file bound geometries
    """
    bounds_gdf = transform_gdf_council_bounds(ladnm_col, use_cols)
    files = base_getters.list_obj_s3_location(path)
    files = [file for file in files if file.endswith(".gml")]

    matches = _match_list_file_to_name(
        land_extent_files=files,
        council_names=bounds_gdf["council_name_std"],
    )

    file_bounds_gdf = pd.DataFrame(
        {"inspire_file_name": files, "council_bounds_matches": matches}
    ).merge(
        bounds_gdf,
        how="left",
        left_on="council_bounds_matches",
        right_on="council_name_std",
    )

    file_bounds_gdf = gpd.GeoDataFrame(
        file_bounds_gdf, crs="EPSG:27700", geometry="geometry"
    )

    use_cols.append("council_name_std")

    file_bounds_gdf = fill_nulls_file_bounds(
        file_bounds_gdf, bounds_gdf, ladnm_col, use_cols
    )

    if any(file_bounds_gdf["geometry"].isna()):
        warnings.warn(
            f"{file_bounds_gdf['geometry'].isna().sum()} land extent INSPIRE .gml files are missing file bound geometries."
        )
    if (file_bounds_gdf["geometry"].isna().sum()) > (len(file_bounds_gdf) * 0.1):
        raise AssertionError(
            f"More than 10% of INSPIRE .gml files are missing file bound geometries.\n"
            f"Please check input INSPIRE .gml and Local Authority bounds files are correct and have sufficient "
            f"geographical coverage."
        )
    return file_bounds_gdf


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
        land_parcels_gdf = load_geodata.load_gdf_inspire_land_parcels(f"s3://{file}")
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
    # TODO - the process to identify nation for processing could be improved
    if "inspire_ew" in land_parcel_file:
        gdf = load_geodata.load_gdf_inspire_land_parcels(
            land_parcel_file, columns=["NATIONALCADASTRALREFERENCE", "geometry"]
        )
    elif "inspire_scotland" in land_parcel_file:
        gdf = load_geodata.load_gdf_inspire_land_parcels(
            land_parcel_file, columns=["nationalca", "geometry"]
        ).rename(columns={"nationalca": "NATIONALCADASTRALREFERENCE"})
    else:
        raise ValueError(
            f"Nation not identified from file path: {land_parcel_file} \n"
            f"Unable to conduct nation-specific preprocessing of land registry file."
        )
    gdf = geo_utils.transform_gdf_drop_duplicates(gdf)
    gdf["land_area_m2"] = gdf["geometry"].area

    return gdf
