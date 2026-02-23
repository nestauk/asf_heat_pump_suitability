import geopandas as gpd
import pandas as pd

from asf_heat_pump_suitability.getters import get_datasets


def load_transform_gdf_lsoa_dz_boundaries() -> gpd.GeoDataFrame:
    """
    Load raw 2021 LSOA/2011 Data Zone geospatial boundary polygons for England and Wales, and Scotland. LSOAs and DataZones
    are combined into single `lsoa` column. CRS British National Grid (EPSG:27700).

    Returns:
        gpd.GeoDataFrame: boundary polygons for 2021 LSOAs / 2011 DataZones
    """
    lsoa_gdf = get_datasets.load_gdf_ons_lsoa_bounds(columns=["LSOA21CD", "geometry"]).rename(
        columns={"LSOA21CD": "lsoa"}
    )
    dz_gdf = get_datasets.load_gdf_scotgov_data_zone_bounds(columns=["DataZone", "geometry"]).rename(
        columns={"DataZone": "lsoa"}
    )
    if lsoa_gdf.crs != dz_gdf.crs:
        raise ValueError(
            f"LSOA and Data Zone GeoDataFrames must share the same CRS, got {lsoa_gdf.crs} and {dz_gdf.crs}"
        )
    return pd.concat([lsoa_gdf, dz_gdf])
