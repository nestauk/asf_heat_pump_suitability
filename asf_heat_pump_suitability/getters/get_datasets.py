import logging

import geopandas as gpd
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(4))
def load_gdf_inspire_land_parcels(path: str, **kwargs) -> gpd.GeoDataFrame:
    """Load land registry's index polygons spatial data (INSPIRE) showing the geometry and
    extent of registered freehold properties in England and Wales. CRS EPSG:27700.

    Args:
        path (str): path to INSPIRE land parcel file
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: registered land extent polygons for one council
    """
    logging.info(f"Loading INSPIRE land parcel file: {path}")
    gdf = gpd.read_file(path, engine="pyogrio", **kwargs)
    return gdf
