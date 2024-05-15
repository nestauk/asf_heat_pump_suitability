import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def get_df_target_nrooms() -> pl.DataFrame:
    """
    Get dataframe of counts of total number of rooms for properties in all LSOAs in England and Wales. Source: census data 2021.

    Returns:
        pl.Dataframe: counts of total number of rooms for properties in all LSOAs in England and Wales
    """
    df = pl.read_csv(config["data_source"]["EW_census_number_of_rooms"])
    df = df.drop(
        [
            "Lower layer Super Output Areas",
            "Number of rooms (Valuation Office Agency) (9 categories)",
        ]
    )
    df = df.rename(
        {
            "Lower layer Super Output Areas Code": "lsoa",
            "Number of rooms (Valuation Office Agency) (9 categories) Code": "number_of_rooms",
        }
    )
    return df.pivot(
        index="lsoa", columns="number_of_rooms", values="Observation"
    ).rename({"9": "9+"})


def get_df_target_property_type(censored_values: int = 0) -> pl.DataFrame:
    """
    Get dataframe of property type counts for all LSOAs in England and Wales, and fill censored values with given
    constant. Source: census data 2021.

    Args:
        censored_values (int): value to fill censored values with, 0-9. Default 0.

    Returns:
        pl.Dataframe: counts of property type for all LSOAs in England and Wales
    """
    content = base_getters.get_content_from_path(
        config["data_source"]["EW_housing_characteristics_census"]
    )
    df = pl.read_excel(content, sheet_name="2c", engine="calamine")

    # Remove empty header rows
    df = df.rename(df[2].to_dicts().pop()).slice(
        3,
    )
    df = df.rename({"Area Code": "lsoa"}).drop(["Area Name"])

    return _fill_df_censored_values(df, censored_values)


def get_df_target_tenure(censored_values: int = 0) -> pl.DataFrame:
    """
    Get dataframe of tenure type counts for all LSOAs in England and Wales, and fill censored values with given
    constant. Source: census data 2021.

    Args:
        censored_values (int): value to fill censored values with, 0-9. Default 0.

    Returns:
        pl.Dataframe: counts of tenure type for all LSOAs in England and Wales
    """
    content = base_getters.get_content_from_path(
        config["data_source"]["EW_housing_characteristics_census"]
    )
    df = pl.read_excel(content, sheet_name="3c", engine="calamine")

    # Remove empty header rows
    df = df.rename(df[2].to_dicts().pop()).slice(
        3,
    )
    df = df.rename(
        {
            "Area Code": "lsoa",
            "Owned or shared ownership": "owner-occupied",
            "Social Rented": "rental (social)",
            "Private Rented or lives rent free": "rental (private)",
        }
    ).drop(["Area Name"])

    return _fill_df_censored_values(df, censored_values)


def get_df_target_tenure_uncensored() -> pl.DataFrame:
    """
    Get dataframe of tenure type counts for all LSOAs in England and Wales. Source: census data 2021.

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
) -> pl.DataFrame:
    """
    Get dataframe of counts of properties built pre- and post-1930 for all LSOAs in England and Wales. Source: Consumer
    Data Research Centre, 2015.

    Args:
        pre_cols (list): columns before age split. Default pre-1930 columns.
        post_cols (list): columns after age split. Default post-1930 columns.

    Returns:
        pl.Dataframe: counts of properties built pre- and post-1930 for all LSOAs in England and Wales.
    """
    df = pl.read_csv(config["data_source"]["EW_cdrc_dwelling_age"])
    df = (
        df.with_columns(
            [
                pl.sum_horizontal(pre_cols).alias("pre"),
                pl.sum_horizontal(post_cols).alias("post"),
            ]
        )
        .rename({"BP_UNKNOWN": "unknown", "AREA_CODE": "lsoa"})
        .select(["lsoa", "pre_1930", "post_1930", "unknown"])
    )

    return df


def _fill_df_censored_values(df: pl.DataFrame, val: int) -> pl.DataFrame:
    """
    Fill censored values in a dataframe with a given value.

    Args:
        df (pl.DataFrame): dataframe
        val (int): value to fill censored values with

    Returns:
        pl.DataFrame: dataframe with filled values
    """
    cols = df.columns
    cols.remove("lsoa")
    return df.with_columns([pl.col(cols).str.replace("c", f"{val}").cast(pl.Int64)])
