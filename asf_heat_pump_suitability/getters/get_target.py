import polars as pl
import warnings

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def get_df_target_nrooms() -> pl.DataFrame:
    """
    Get dataframe of counts of total number of rooms for properties in all LSOAs in England and Wales. Where number of rooms
    is >= 9, counts are combined into `9+` category. Source: census data 2021.

    Returns:
        pl.Dataframe: counts of total number of rooms for properties in all LSOAs in England and Wales
    """
    df = pl.read_csv(config["data_source"]["EW_census_number_of_rooms"])
    df = (
        df.drop(
            [
                "Lower layer Super Output Areas",
                "Number of rooms (Valuation Office Agency) (9 categories)",
            ]
        )
        .rename(
            {
                "Lower layer Super Output Areas Code": "lsoa",
                "Number of rooms (Valuation Office Agency) (9 categories) Code": "number_of_rooms",
            }
        )
        .pivot(index="lsoa", columns="number_of_rooms", values="Observation")
        .rename({"9": "9+"})
    )

    return df


def get_df_target_property_type(fill_censored: int = 0) -> pl.DataFrame:
    """
    Get dataframe of property type counts for all LSOAs in England and Wales, and fill censored values (counts below 10)
    with given constant. Source: census data 2021.

    Args:
        fill_censored (int): value to fill censored values with, [0-10]. Default 0.

    Returns:
        pl.Dataframe: counts of property type for all LSOAs in England and Wales
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_housing_characteristics"]
    )
    df = pl.read_excel(content, sheet_name="2c", engine="calamine")

    # Remove empty header rows
    df = (
        df.rename(df[2].to_dicts().pop())
        .slice(
            3,
        )
        .drop(["Area Name"])
        .rename({"Area Code": "lsoa"})
    )
    df = _fill_df_censored_values(df, fill_censored)

    return df


def get_df_target_tenure(fill_censored: int = 0) -> pl.DataFrame:
    """
    Get dataframe of tenure type counts for all LSOAs in England and Wales, and fill censored values (counts below 10)
    with given constant. Source: census data 2021.

    Args:
        fill_censored (int): value to fill censored values with, [0-10]. Default 0.

    Returns:
        pl.Dataframe: counts of tenure type for all LSOAs in England and Wales
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_housing_characteristics"]
    )
    df = pl.read_excel(content, sheet_name="3c", engine="calamine")

    # Remove empty header rows
    df = (
        df.rename(df[2].to_dicts().pop())
        .slice(
            3,
        )
        .drop(["Area Name"])
        .rename(
            {
                "Area Code": "lsoa",
                "Owned or shared ownership": "owner-occupied",
                "Social Rented": "rental (social)",
                "Private Rented or lives rent free": "rental (private)",
            }
        )
    )

    df = _fill_df_censored_values(df, fill_censored)

    return df


def get_df_target_tenure_uncensored() -> pl.DataFrame:
    """
    Get dataframe of tenure type counts for all LSOAs in England and Wales. Dataframe has no censored values. Source:
    census data 2021.

    Returns:
        pl.Dataframe: counts of tenure type for all LSOAs in England and Wales
    """
    df = pl.read_csv(config["data_source"]["EW_census_tenure"])

    owned_cols = [
        "Owned: Owns with a mortgage or loan",
        "Owned: Owns outright",
        "Shared ownership: Shared ownership",
    ]

    social_rent_cols = [
        "Social rented: Other social rented",
        "Social rented: Rents from council or Local Authority",
    ]

    private_rent_cols = [
        "Lives rent free",
        "Private rented: Private landlord or letting agency",
        "Private rented: Other private rented",
    ]

    df = (
        df.rename(
            {
                "Lower layer Super Output Areas Code": "lsoa",
                "Tenure of household (9 categories)": "tenure",
            }
        )
        .pivot(index="lsoa", columns="tenure", values="Observation")
        .with_columns(
            [
                pl.sum_horizontal(owned_cols).alias("owner-occupied"),
                pl.sum_horizontal(social_rent_cols).alias("rental (social)"),
                pl.sum_horizontal(private_rent_cols).alias("rental (private)"),
            ]
        )
        .select(
            pl.col(["lsoa", "owner-occupied", "rental (social)", "rental (private)"])
        )
    )

    return df


def get_df_target_build_year(
    pre_cols: list = config["mapping"]["build_year_pre_cols"],
    post_cols: list = config["mapping"]["build_year_post_cols"],
    year_label: int = 1930,
) -> pl.DataFrame:
    """
    Get dataframe of counts of properties built before and after given year (determined by `pre_cols` and `post_cols`
    params) for all LSOAs in England and Wales. Source: Consumer Data Research Centre, 2015.

    Args:
        pre_cols (list): column names of age bands before age split. Default pre-1930 columns.
        post_cols (list): column names of age bands after age split. Default post-1930 columns.
        year_label (int): build year when split occurs

    Returns:
        pl.Dataframe: counts of properties built before and after given year for all LSOAs in England and Wales.
    """
    df = pl.read_csv(config["data_source"]["EW_cdrc_dwelling_age"])
    df = (
        df.with_columns(
            [
                pl.sum_horizontal(pre_cols).alias(f"pre_{year_label}"),
                pl.sum_horizontal(post_cols).alias(f"post_{year_label}"),
            ]
        )
        .rename({"BP_UNKNOWN": "unknown", "AREA_CODE": "lsoa"})
        .select(["lsoa", f"pre_{year_label}", f"post_{year_label}", "unknown"])
    )

    return df


def _fill_df_censored_values(df: pl.DataFrame, val: int) -> pl.DataFrame:
    """
    Fill censored values in a target dataframe with a given value.

    Args:
        df (pl.DataFrame): dataframe
        val (int): value to fill censored values with, [0-10]

    Returns:
        pl.DataFrame: dataframe with filled values
    """
    if not (0 <= val <= 10):
        warnings.warn(
            "Value to fill censored target data should be within range [0-10]. "
            "Values outside this range may significantly change target proportions."
        )
    cols = df.columns
    cols.remove("lsoa")
    df = df.with_columns([pl.col(cols).str.replace("c", f"{val}").cast(pl.Int64)])

    return df
