import polars as pl
import geopandas as gpd
import regex as re
import os

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def load_df_osopen_uprn(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid X and Y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OSOpen UPRNs...")
    path = config["data"]["geodata"]["uk_osopen_uprn"]
    filename = os.path.basename(path).split("_csv")[0]
    df = base_getters.get_df_from_zip_csv_s3(
        path,
        extract_file=f"{filename}.csv",
        **kwargs,
    )

    return df


def load_gdf_heat_network_zones(local_authority: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with heat network zone polygons in given Local Authority.

    Args:
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: polygons of heat network zones in given Local Authority.
    """

    local_authority = local_authority.lower()

    if local_authority not in config["data"]["geodata"]["heat_network_zones"].keys():
        raise ValueError(
            f"No path found for heat network zone geodata in Local Authority: {local_authority}"
        )

    gdf = base_getters.get_gdf_from_gpkg_s3_path(
        path=config["data"]["geodata"]["heat_network_zones"][local_authority],
        **kwargs,
    )

    print(
        f"Heat network zone geodataframe successfully loaded for {local_authority} with CRS {gdf.crs}."
    )
    return gdf


def load_gdf_spatial_signatures_gb(
    detail_level: str = "simplified", **kwargs
) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with polygons in GB from the Spatial Signatures Framework  classified by their
    Spatial Signature type. (Source: https://doi.org/10.6084/m9.figshare.16691575).

    The dataset can be loaded at two levels of detail:
    - "simplified": geometries provided with only fields "id" and "type"
    - "full": geometries provided with all fields

    Args:
        detail_level (str, optional): Which level of descriptive detail to load.
            Must be either "simplified" or "full". Defaults to "simplified".

    Returns:
        gpd.GeoDataFrame: spatial signature polygons in GB.
    """

    if detail_level not in {"full", "simplified"}:
        raise ValueError(
            f"detail_level must be 'full' or 'simplified', not {detail_level}"
        )

    gdf = base_getters.get_gdf_from_gpkg_s3_path(
        path=config["data"]["geodata"]["gb_spatial_signatures"][detail_level],
        **kwargs,
    )

    print(
        f"Spatial signatures {detail_level} geodataframe successfully loaded with CRS {gdf.crs}."
    )

    return gdf
