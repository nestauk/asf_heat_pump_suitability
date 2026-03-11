import geopandas as gpd

from asf_heat_pump_suitability import config


def verify_gdf_crs(gdf: gpd.GeoDataFrame, target_crs: str | int = None) -> gpd.GeoDataFrame:
    """Verify GeoDataFrame uses the defined target CRS. If not, convert to target CRS.

    Args:
        gdf (gpd.GeoDataFrame): geodataset to check CRS of
        target_crs (str|int): target CRS. Defaults to target CRS defined in config base.yaml

    Returns:
        gpd.GeoDataFrame: geodataset in target CRS
    """
    if target_crs is None:
        target_crs = config["constant"]["target_crs"]
    if gdf.crs != target_crs:
        print(f"Converting GeoDataFrame to target CRS: {target_crs}")
        return gdf.to_crs(target_crs)
    return gdf
