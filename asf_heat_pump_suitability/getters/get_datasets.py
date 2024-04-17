import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas


def get_df_onspd_gb(
    pcd_col: str = "pcd",
    use_cols: list = ["pcd", "lsoa11", "msoa11", "lsoa21", "msoa21"],
) -> pl.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args
        pcd_col (str): name of column containing postcodes. Default `"pcd"`.
        use_cols (list): columns to import. Default `["pcd", "lsoa11", "msoa11", "lsoa21", "msoa21"]`.

    Returns
        pl.DataFrame: postcode directory for Great Britain
    """
    df = base_getters.get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        schema=schemas.onspd_schema,
        columns=use_cols,
    )

    df = df.with_columns(
        pl.col(pcd_col).str.to_uppercase().str.replace(" ", "").alias("postcode")
    )

    return df
