import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas


def get_df_ons_pd(**kwargs) -> pl.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: postcode directory for Great Britain
    """
    df = base_getters.get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        schema=schemas.onspd_schema,
        **kwargs,
    )

    return df


def get_df_ons_garden_space_avg(**kwargs) -> pl.DataFrame:
    """
    Get raw ONS 'Access to garden space, Great Britain' dataset.

    Args:
        **kwargs for pl.read_excel

    Returns:
        pl.DataFrame: raw ONS 'Access to garden space' dataset
    """
    content = base_getters.get_content_from_path(
        config["data_source"]["GB_ons_garden_space_access"]
    )
    df = pl.read_excel(
        content,
        sheet_name="MSOA gardens",
        engine="calamine",
        **kwargs,
    )
    return df
