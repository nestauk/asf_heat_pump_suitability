import geopandas as gpd
from asf_heat_pump_suitability import config

stoke_ward_boundary = (
    "s3://asf-heat-pump-suitability/source_data/stoke_ward_boundary.geojson"
)
SX_Greenspace = (
    "s3://asf-heat-pump-suitability/source_data/SX_Greenspace/SX_GreenspaceSite.shp"
)


def load_stoke_bound(**kwargs) -> gpd.GeoDataFrame:
    """
    Load Stoke ward boundary from 2025-03-04 downloaded from https://www.planning.data.gov.uk/entity/800351#geojson
    (CRS: EPSG:4326).

    Args:
        **kwargs for gpd.read_file

    Returns:
        gpd.GeoDataFrame: geography for Stoke ward
    """
    gdf = gpd.read_file(stoke_ward_boundary, **kwargs)

    return gdf


def load_SX_Greenspace(**kwargs) -> gpd.GeoDataFrame:
    """
    Load the OS open greenspace data for the SX region of the UK https://osdatahub.os.uk/downloads/open/OpenGreenspace

    Args:
        **kwargs for gpd.read_file

    Returns:
        gpd.GeoDataFrame: polygons of greenspaces in the Plymouth area and beyond
    """
    gdf = gpd.read_file(SX_Greenspace, **kwargs)

    return gdf
