import geopandas as gpd
import shapely
from pygeotile import tile
import pyproj
import warnings
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.utils import geo_utils


def transform_df_uk_dataset_links():
    """ """
    df = get_datasets.load_df_microsoft_building_footprint_links()
    df = df[df["Location"] == "UnitedKingdom"]

    transformer = _set_crs_transformer()

    df["ms_bbox_BNG"] = [
        convert_quadkey_to_bng_bounds(qk, transformer) for qk in df["QuadKey"]
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


def convert_quadkey_to_bng_bounds(quadkey, transformer):
    """ """
    min_latlon, max_latlon = tile.Tile.from_quad_tree(quadkey).bounds

    min_lat = min_latlon.latitude
    min_lon = min_latlon.longitude
    max_lat = max_latlon.latitude
    max_lon = max_latlon.longitude

    min_lonlat = transformer.transform(min_lon, min_lat)
    max_lonlat = transformer.transform(max_lon, max_lat)

    return min_lonlat, max_lonlat


def convert_bounds_to_points(min_lonlat, max_lonlat):
    """
    CRS: British National Grid
    """
    min_lat = min_lonlat[1]
    min_lon = min_lonlat[0]
    max_lat = max_lonlat[1]
    max_lon = max_lonlat[0]

    bbox_points = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]

    return bbox_points


def _set_crs_transformer(from_crs="epsg:4326", to_crs="epsg:27700"):
    """ """
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)

    return transformer


def transform_gdf_building_footprints(ms_file):
    """
    Add ID and drop duplicates
    """
    gdf = get_datasets.load_gdf_microsoft_building_footprints(ms_file)
    gdf = extend_gdf_building_footprint_id(gdf)
    gdf = geo_utils.transform_gdf_drop_close_duplicates(gdf)
    if len(gdf["building_id"].nunique) != len(gdf):
        warnings.warn(
            f"There are building footprint polygons with duplicate IDs in file: {ms_file}"
        )
    gdf["building_area"] = gdf["geometry"].area

    # TODO: do we want to drop building footprints below a certain confidence score? The dataset has confidence score
    # TODO: for some but not all footprints. Not sure how many it's available for, might be a low number

    return gdf


def extend_gdf_building_footprint_id(gdf):
    """
    Add replicable unique ID column to building footprint gdf. Use representative point
    """
    coords = gdf.representative_point().get_coordinates()
    ids = coords["x"].astype(str) + "_" + coords["y"].astype(str)
    gdf["building_id"] = ids

    return gdf
