import geopandas as gpd

TARGET_CRS = 27700  # British National Grid


def verify_gdf_crs(gdf: gpd.GeoDataFrame, target_crs: str | int = None) -> gpd.GeoDataFrame:
    """Verify GeoDataFrame uses the defined target CRS. If not, convert to target CRS.

    Args:
        gdf (gpd.GeoDataFrame): geodataset to check CRS of
        target_crs (str|int): target CRS. Defaults to TARGET_CRS (British National Grid, 27700)

    Returns:
        gpd.GeoDataFrame: geodataset in target CRS
    """
    if target_crs is None:
        target_crs = TARGET_CRS
    if gdf.crs != target_crs:
        print(f"Converting GeoDataFrame to target CRS: {target_crs}")
        return gdf.to_crs(target_crs)
    return gdf
