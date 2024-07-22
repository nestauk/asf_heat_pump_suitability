import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas
from io import StringIO


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
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["GB_ons_garden_space_access"]
    )
    df = pl.read_excel(
        content,
        sheet_name="MSOA gardens",
        engine="calamine",
        **kwargs,
    )
    return df


def get_df_osopen_uprn_latlon(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and x and y coordinates for all
    UPRNs in Great Britain.

    Args:
        **kwargs fo pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    df = base_getters.get_df_from_zip_csv_s3(
        config["data_source"]["GB_osopen_uprn_latlon"],
        extract_file="osopenuprn_202405.csv",
        **kwargs,
    )

    return df


def get_df_ons_number_of_households() -> pl.DataFrame:
    """
    Get raw ONS 'Number of households' dataset.

    Args:
        **kwargs for pl.read_excel

    Returns:
        pl.DataFrame: raw ONS 'Number of households' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_number_of_households"],
    )
    content_str = content.decode("utf-8")  # convert bytes to string
    content_file = StringIO(content_str)  # convert string to file-like object
    df = pl.read_csv(content_file, skip_rows=6, has_header=True)
    # Preprocessing steps due to white space
    # Remove the last eight rows
    df = df.slice(0, len(df) - 9)
    # Remove the first row
    df = df.slice(1, len(df) - 1)
    return df


def get_df_ons_land_area() -> pl.DataFrame:
    """
    Get raw ONS 'land area' dataset.

    Returns:
        pl.DataFrame: raw ONS 'land area' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_land_area"],
    )
    content_str = content.decode("utf-8")  # convert bytes to string
    content_file = StringIO(content_str)  # convert string to file-like object

    # dtypes specificed as polars read csv wa inferring wrong data types and throwing error
    dtypes = {
        "LSOA21CD": pl.Utf8,
        "LSOA21NM": pl.Utf8,
        "Extent of the Realm (Area in KM2)": pl.Float64,
        "Clipped to the Coastline (Area in KM2)": pl.Float64,
        "Area of Inland Water (KM2)": pl.Float64,
        "Land Count (Area in KM2)": pl.Float64,
        "LTLA22CD": pl.Utf8,
        "LTLA22NM": pl.Utf8,
        "LTLA22NMW": pl.Utf8,
    }
    df = pl.read_csv(content_file, dtypes=dtypes, has_header=True)
    return df


def get_df_spa_offgasgrid() -> pl.DataFrame:
    """
    Get off gas grid data from Supply Point Administration dataset
    Returns:
        pl.DataFrame: raw off gas grid dataset
    """
    df = base_getters.get_df_from_excel_s3_path(
        config["data_source"]["UK_spa_offgasgrid"], sheet_name="Off-Gas Postcodes 2024"
    )
    return df


get_df_spa_offgasgrid()
