import logging
from typing import Union, Optional, List

import s3fs

import pandas as pd

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry
from shapely import wkb

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils


def transform_gdf_drop_duplicates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drop polygons with the same representative point. This drops both duplicate and nearly identical geometries.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with duplicate polygon geometries dropped
    """
    gdf["rep_point"] = gdf.representative_point().to_wkb()
    if gdf["rep_point"].nunique() != len(gdf):
        duplicate_count = gdf.duplicated(subset="rep_point").sum()

        logging.info(
            f"Polygons containing same representative point found. "
            f"Dropping {duplicate_count} polygons."
        )

        # Sort values to create replicable duplicate removal process
        gdf = gdf.sort_values(by="geometry").drop_duplicates(
            subset="rep_point", keep="first"
        )

    gdf = gdf.drop("rep_point", axis=1)

    return gdf


def get_polygon_gdf_bounds(gdf: gpd.GeoDataFrame) -> shapely.Polygon:
    """
    Get bounding polygon of GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame

    Returns:
        shapely.Polygon: bounding polygon of GeoDataFrame
    """
    return shapely.box(*gdf.total_bounds)


def parse_binary_geometry(
    binary_data: Union[BaseGeometry, bytes, str],
) -> Union[BaseGeometry, None]:
    """
    Parse binary geometry data into Shapely geometry object.

    Args:
        binary_data: Input geometry data in various formats.

    Returns:
        Shapely geometry object or None if parsing fails.
    """
    if isinstance(binary_data, BaseGeometry):
        return binary_data
    elif isinstance(binary_data, bytes):
        return wkb.loads(binary_data)
    elif isinstance(binary_data, str):
        return wkb.loads(binary_data, hex=True)
    else:
        return None


def verify_gdf_crs(
    gdf: gpd.GeoDataFrame, target_crs: str | int = config["constant"]["target_crs"]
) -> gpd.GeoDataFrame:
    """
    Verify GeoDataFrame uses the defined target CRS. If not, convert to target CRS.

    Args:
        gdf (gpd.GeoDataFrame): geodataset to check CRS of
        target_crs (str|int): target CRS. Defaults to target CRS defined in config base.yaml

    Returns:
        gpd.GeoDataFrame: geodataset in target CRS
    """
    if gdf.crs != target_crs:
        print(f"Converting GeoDataFrame to target CRS: {target_crs}")
        return gdf.to_crs(target_crs)
    else:
        return gdf


def find_gdf_overlapping_geometries(
    gdf: gpd.GeoDataFrame, return_geometries: bool = True, id_col=None
) -> gpd.GeoDataFrame | set:
    """
    Find overlapping geometries within the same GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): containing geometries to check for overlaps
        return_geometries (bool): set to `True` to return a `GeoDataFrame` with the overlapping geometries. Set to `False`
        to return just the IDs of overlapping geometries.
        id_col (str): name of unique ID for geometries. Required only when `return_geometries` is set to `False`. Default
        `None` to use index.

    Returns:
        gpd.GeoDataFrame | set: geometries or IDs of geometries which overlap with any other geometry within the `GeoDataFrame`
    """
    buffer = gdf.copy()
    # Buffer added to prevent touching neighbours being identified as overlapping
    buffer["geometry"] = buffer["geometry"].buffer(-0.0001)
    intersecting_gdf = buffer.sjoin(buffer, predicate="intersects")
    intersecting_gdf = intersecting_gdf.loc[
        intersecting_gdf.index != intersecting_gdf.index_right
    ]
    if return_geometries:
        return intersecting_gdf
    else:
        if id_col:
            overlapping_ids = set(intersecting_gdf[f"{id_col}_left"])
        else:
            overlapping_ids = set(intersecting_gdf.index)
            id_col = "index"

        print(f"ID used: {id_col}. Overlapping ID count: {len(overlapping_ids)}")
        return overlapping_ids


def map_dict_files_to_boundaries(dir_path: str, save_as: str = None) -> dict:
    """
    Create mapping of geospatial file names in a given S3 directory to their boundaries.

    Args:
        dir_path (str): path to S3 directory containing geospatial files of interest.
        save_as (str): Optional. Save the mapping to a geospatial file type. Default None which does not save the output.

    Returns:
        dict: mapping of file names to boundary geometries
    """
    mapping = dict()
    geo_files = list_geo_files(dir_path)
    print(f"Found {len(geo_files)} files to map. Beginning mapping...")
    for file in geo_files:
        print(f"\nLoading: {file}")
        gdf = gpd.read_file(f"s3://{file}").to_crs(epsg=27700)
        gdf["geometry"] = gdf["geometry"].make_valid()
        # Dissolving all geometries in gdf into one
        mapping[file] = gdf.dissolve()["geometry"].iloc[0]

    if save_as:
        gdf = gpd.GeoDataFrame(
            {"file_path": mapping.keys(), "geometry": mapping.values()},
            geometry="geometry",
            crs=27700,
        )
        save_utils.save_to_s3(gdf, save_as)

    return mapping


def list_geo_files(dir_path: str) -> list:
    """
    List all geospatial files in a directory on S3 of types geojson, geopackage, and shapefiles.

    Args:
        dir_path (str): S3 URI to directory

    Returns:
        list: S3 URIs to all geospatial files in directory
    """
    extensions = ["geojson", "gpkg", "shp"]
    fs = s3fs.S3FileSystem()
    dir_path = dir_path.rstrip("/")
    return [f for ext in extensions for f in fs.glob(f"{dir_path}/*.{ext}")]


def concat_gdfs(
    dir_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    gdfs: Optional[List[gpd.GeoDataFrame]] = None,
    crs: int = 27700,
    save_as: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Concatenate list of geodataframes from a single given source (e.g. directory, list of file paths, list of
    geodataframes) into a single one.

    Args:
        dir_path (str): path to S3 directory containing files of interest to concatenate. Optional.
        file_paths (List[str]): list of file paths containing data to concatenate. Optional.
        gdfs (List[gpd.GeoDataFrame]): list of geodataframes to concatenate. Optional.
        crs (str): CRS of final geodataframe. Default 27700 (BNG).
        save_as (str): Optional. Save the mapping to a geospatial file type. Default None which does not save the output.

    Returns:
        gpd.GeoDataFrame: concatenated geodataframe
    """
    if not any([dir_path, file_paths, gdfs]):
        raise ValueError("One of `dir_path`, `file_paths`, or `gdfs` required.")
    if sum([arg is not None for arg in [dir_path, file_paths, gdfs]]) != 1:
        raise ValueError(
            "Please select only one of `dir_path`, `file_paths`, or `gdfs`."
        )

    if any([dir_path, file_paths]):
        if dir_path:
            file_paths = list_geo_files(dir_path)
            print(f"Found {len(file_paths)} files to concatenate.")

        gdfs = []
        for file in file_paths:
            print(f"\nLoading: {file}")
            gdf = gpd.read_file(f"s3://{file}").to_crs(epsg=crs)
            gdf["geometry"] = gdf["geometry"].make_valid()
            gdfs.append(gdf)

        concat_gdf = pd.concat(gdfs).set_geometry("geometry")
    else:
        concat_gdf = pd.concat([gdf.to_crs(epsg=crs) for gdf in gdfs]).set_geometry(
            "geometry"
        )

    if save_as:
        save_utils.save_to_s3(concat_gdf, save_as)

    return concat_gdf
