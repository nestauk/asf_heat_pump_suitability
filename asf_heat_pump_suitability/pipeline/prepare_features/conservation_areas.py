import geopandas as gpd
import pandas as pd

from asf_heat_pump_suitability.getters import get_datasets


def transform_gdf_conservation_areas_england() -> gpd.GeoDataFrame:
    """
    Load and transform Historic England conservation areas to CRS EPSG:27700.

    Returns:
        gpd.GeoDataFrame: Historic England conservation areas in CRS EPSG:27700
    """
    gdf = get_datasets.load_gdf_historic_england_conservation_areas()[
        ["name", "geometry"]
    ]

    gdf = gdf.to_crs("EPSG:27700").rename(columns={"name": "in_conservation_area"})

    return gdf


def generate_gdf_conservation_areas_england_lad(
    ladcd_col: str = "LAD23CD",
) -> pd.DataFrame:
    """
    Generate dataframe of UK local authority districts (LADs) with indicator of conservation area data availability.

    Args:
        ladcd_col (str): name of column in local authority district (LAD) boundaries file with LAD codes

    Returns:
        pd.DataFrame: conservation area data availability per LAD in the UK
    """
    conservation_areas_gdf = transform_gdf_conservation_areas_england()

    council_bounds = get_datasets.load_gdf_ons_council_bounds().to_crs(epsg="27700")

    # Join conservation areas to their councils
    lad_conservation_areas_gdf = council_bounds.sjoin(
        conservation_areas_gdf, how="left", predicate="intersects"
    )[[ladcd_col, "in_conservation_area"]]

    lad_conservation_areas_gdf = lad_conservation_areas_gdf.groupby("LAD23CD").agg(
        {"in_conservation_area": "count"}
    )
    lad_conservation_areas_gdf["lad_conservation_area_data_available"] = (
        lad_conservation_areas_gdf["in_conservation_area"].astype(bool)
    )
    lad_conservation_areas_gdf = lad_conservation_areas_gdf.drop(
        columns=["in_conservation_area"]
    )

    return lad_conservation_areas_gdf
