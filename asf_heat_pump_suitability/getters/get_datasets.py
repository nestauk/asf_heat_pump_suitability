import polars as pl
import geopandas as gpd
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas


def get_df_ons_pd(**kwargs) -> pl.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args
        **kwargs for pl.read_csv

    Returns
        pl.DataFrame: postcode directory for Great Britain
    """
    df = base_getters.get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        schema=schemas.onspd_schema,
        **kwargs,
    )

    return df


def load_gdf_ons_council_bounds():
    """ """
    gdf = base_getters.load_gdf_from_s3_geojson(
        config["data_source"]["UK_ons_lad_bounds"]
    )

    return gdf
