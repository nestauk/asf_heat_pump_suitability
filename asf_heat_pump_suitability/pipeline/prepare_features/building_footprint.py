import geopandas as gpd
import shapely
from pygeotile import tile
import pyproj
import warnings
from convertbng.util import convert_bng
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.utils import geo_utils


def transform_df_uk_dataset_links() -> gpd.GeoDataFrame:
    """
    Load dataset containing URLs to Microsoft Building Footprint files for the UK only. Get bounding
    geometry of files in CRS: EPSG:27700, British National Grid.

    Returns:
        gpd.GeoDataFrame: URLs to Microsoft building footprint files for UK only with bounding geometries of files
    """
    df = get_datasets.load_df_microsoft_building_footprint_links()
    df = df[df["Location"] == "UnitedKingdom"]

    transformer = _set_crs_transformer()

    # Use convertbng to convert QuadKeys where possible, otherwise use pyproj transformer
    df["geometry"] = [
        (
            polygons
            if (polygons := convertbng_quadkey_to_polygon(qk))
            else convertpyproj_quadkey_to_polygon(qk, transformer)
        )
        for qk in df["QuadKey"]
    ]

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:27700").rename(
        columns={"Url": "ms_url"}
    )

    return gdf


def convertbng_quadkey_to_polygon(quadkey: str) -> shapely.Polygon:
    """
    Convert Microsoft QuadKey (QuadTree Key) to polygon in British National Grid (CRS: EPSG:27700) using convertbng for
    conversion accuracy up to 1.1mm.

    Args:
        quadkey (str): Microsoft QuadKey

    Returns:
        shapely.Polygon
    """
    minlatlon, maxlatlon = tile.Tile.from_quad_tree(quadkey).bounds
    x, y = convert_bng(
        [minlatlon.longitude, maxlatlon.longitude],
        [minlatlon.latitude, maxlatlon.latitude],
    )
    return shapely.box(xmin=x[0], ymin=y[0], xmax=x[1], ymax=y[1])


def convertpyproj_quadkey_to_polygon(
    quadkey: str, transformer: pyproj.Transformer
) -> shapely.Polygon:
    """
    Convert Microsoft QuadKey (QuadTree Key) to polygon in British National Grid (CRS: EPSG:27700) using pyproj for
    conversion accuracy up to 5m.

    Args:
        quadkey (str): Microsoft QuadKey
        transformer (pyproj.Transformer): transformer to transform points between coordinate systems

    Returns:
        shapely.Polygon
    """
    min_latlon, max_latlon = tile.Tile.from_quad_tree(quadkey).bounds

    min_xy = transformer.transform(min_latlon.longitude, min_latlon.latitude)
    max_xy = transformer.transform(max_latlon.longitude, max_latlon.latitude)

    return shapely.box(xmin=min_xy[0], ymin=min_xy[1], xmax=max_xy[0], ymax=max_xy[1])


def _set_crs_transformer(
    from_crs: str = "epsg:4326", to_crs: str = "epsg:27700"
) -> pyproj.Transformer:
    """
    Set Coordinate Reference System (CRS) transformer to convert between coordinate systems. Function will accept as
    input and return as output coordinates using the traditional GIS order, that is longitude, latitude for geographic
    CRS and easting, northing for most projected CRS.

    Args:
        from_crs (str): CRS to convert from
        to_crs (str): CRS to convert to

    Returns:
        pyproj.Transformer: transformer to transform points between coordinate systems
    """
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)

    return transformer


def transform_gdf_building_footprints(building_footprint_file: str) -> gpd.GeoDataFrame:
    """
    Load and transform building footprints dataframe. Generate unique ID for each building, drop duplicate
    geometries, and get building area (m2) for each building polygon. CRS: EPSG:27700, British National Grid (BNG).

    Args:
        building_footprint_file (str): URL of Microsoft building footprints file

    Returns:
        gpd.GeoDataFrame: building footprints polygons in BNG with unique IDs, and footprint area in m2
    """
    gdf = get_datasets.load_gdf_microsoft_building_footprints(building_footprint_file)
    gdf["geometry"] = transform_geoseries_convert_bng(gdf["geometry"])
    gdf = extend_gdf_building_footprint_id(gdf)
    gdf = geo_utils.transform_gdf_drop_duplicates(gdf)
    if gdf["building_id"].nunique != len(gdf):
        warnings.warn(
            f"There are building footprint polygons with duplicate IDs in file: {building_footprint_file}"
        )
    gdf["building_area_m2"] = gdf["geometry"].area

    # TODO: do we want to drop building footprints below a certain confidence score? The dataset has confidence score
    # TODO: for some but not all footprints. Not sure how many it's available for, might be a low number

    return gdf


def transform_geoseries_convert_bng(geos: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Transform GeoSeries of shapely.Polygon objects in CRS WGS84 to GeoSeries of shapely polygons in CRS EPSG:27700
    (British National Grid) with OSTN15 adjustments for conversion accuracies within 1.1mm.

    Args:
        geos (gpd.GeoSeries): shapely polygons in CRS WGS84

    Returns:
        gpd.GeoSeries: shapely polygons in CRS EPSG:27700 (British National Grid)
    """
    coords = geos.get_coordinates()
    coords["x"], coords["y"] = convert_bng(coords["x"], coords["y"])
    # TODO: conversion back to polygons is rate-limiting step
    s = coords.groupby(coords.index).apply(lambda l: shapely.Polygon(zip(l.x, l.y)))
    return gpd.GeoSeries(s).set_crs(epsg=27700)


def extend_gdf_building_footprint_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign replicable unique ID to building footprint polygons using representative point.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with building footprint polygons

    Returns:
        gpd.GeoDataFrame: building footprint GeoDataFrame with unique ID column
    """
    coords = gdf.representative_point().get_coordinates()
    ids = coords["x"].astype(str) + "_" + coords["y"].astype(str)
    gdf["building_id"] = ids

    return gdf
