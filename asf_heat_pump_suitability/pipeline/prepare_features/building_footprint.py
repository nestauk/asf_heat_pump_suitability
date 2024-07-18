import geopandas as gpd
import shapely
from pygeotile import tile
import pyproj
import warnings
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

    df["ms_bbox_BNG"] = [
        convert_quadkey_to_bounds(qk, transformer) for qk in df["QuadKey"]
    ]
    df["ms_bbox_points"] = [
        convert_bounds_to_points(bbox[0], bbox[1]) for bbox in df["ms_bbox_BNG"]
    ]
    df["geometry"] = [
        shapely.Polygon(bbox_points) for bbox_points in df["ms_bbox_points"]
    ]

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:27700").rename(
        columns={"Url": "ms_url"}
    )

    return gdf


def convert_quadkey_to_bounds(quadkey: str, transformer: pyproj.Transformer) -> tuple:
    """
    Convert Microsoft QuadKey (QuadTree Key) to bounds.

    Args:
        quadkey (str): Microsoft QuadKey
        transformer (pyproj.Transformer): transformer to transform points between coordinate systems

    Returns:
        tuple: min x,y coordinates, and max x,y coordinates of QuadKey
    """
    min_latlon, max_latlon = tile.Tile.from_quad_tree(quadkey).bounds

    min_lat = min_latlon.latitude
    min_lon = min_latlon.longitude
    max_lat = max_latlon.latitude
    max_lon = max_latlon.longitude

    min_xy = transformer.transform(min_lon, min_lat)
    max_xy = transformer.transform(max_lon, max_lat)

    return min_xy, max_xy


def convert_bounds_to_points(min_xy: tuple, max_xy: tuple) -> list:
    """
    Convert bounds to bounding points.

    Args:
        min_xy (tuple): minimum x, y coordinates of bounds
        max_xy (tuple): maximum x, y coordinates of bounds

    Returns:
        list: points of bounding box
    """
    minx = min_xy[0]
    miny = min_xy[1]
    maxx = max_xy[0]
    maxy = max_xy[1]

    bbox_points = [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny],
    ]

    return bbox_points


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
    geometries, and get building area (m2) for each building polygon. CRS: EPSG:27700, British National Grid.

    Args:
        building_footprint_file (str): URL of Microsoft building footprints file

    Returns:
        gpd.GeoDataFrame: building footprints with unique IDs and area in m2
    """
    gdf = get_datasets.load_gdf_microsoft_building_footprints(building_footprint_file)
    gdf = gdf.to_crs("EPSG:27700")
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
