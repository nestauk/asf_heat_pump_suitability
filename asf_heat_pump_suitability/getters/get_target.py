import polars as pl
import polars.selectors as cs

from asf_heat_pump_suitability import config


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


def transform_df_target_property_type() -> pl.DataFrame:
    """
    Load and transform property type counts per LSOA/data zone for England, Scotland, and Wales from census data.

    Returns:
        pl.DataFrame: property type counts for England, Scotland, and Wales per LSOA
    """
    ew_df = load_transform_df_target_property_type_ew()
    s_df = load_transform_df_target_property_type_scotland()
    s_df = s_df.select(ew_df.columns)

    return pl.concat([ew_df, s_df], how="vertical")


def load_transform_df_target_property_type_ew() -> pl.DataFrame:
    """
    Get dataframe of property type counts for all LSOAs in England and Wales from census data.

    Returns:
        pl.Dataframe: counts of property type for all LSOAs in England and Wales
    """
    df = pl.read_csv(config["data_source"]["EW_census_accommodation_type"])
    df = (
        df.select(
            [
                "Lower layer Super Output Areas Code",
                "Accommodation type (8 categories)",
                "Observation",
            ]
        )
        .rename({"Lower layer Super Output Areas Code": "lsoa"})
        .pivot(
            index="lsoa",
            columns="Accommodation type (8 categories)",
            values="Observation",
        )
        .with_columns(
            pl.sum_horizontal(
                [
                    "In a purpose-built block of flats or tenement",
                    "Part of another converted building, for example, former school, church or warehouse",
                    "In a commercial building, for example, in an office building, hotel or over a shop",
                ]
            ).alias("Flat, maisonette or apartment")
        )
        .select(
            [
                "lsoa",
                "Detached",
                "Semi-detached",
                "Terraced",
                "Flat, maisonette or apartment",
                "A caravan or other mobile or temporary structure",
            ]
        )
        .rename(
            {
                "Terraced": "Terraced (including end-terrace)",
                "A caravan or other mobile or temporary structure": "Caravan or other mobile or temporary structure",
            }
        )
    )

    return df


def load_transform_df_target_property_type_scotland() -> pl.DataFrame:
    """
    Load and transform dataframe of property type counts for data zones in Scotland from census data.

    Returns:
        pl.Dataframe: counts of property type for all data zones in Scotland
    """
    df = pl.read_csv(
        config["data_source"]["S_census_accommodation_type"],
        skip_rows=10,
        columns=list(range(0, 11)),
        infer_schema_length=10000,
    )
    df = (
        df[1:]
        .drop_nulls(subset=cs.numeric())
        .drop(
            [
                "Whole house or bungalow: Total",
                "Flat, maisonette or apartment: Total",
                "All occupied households",
            ]
        )
    )
    flats_cols = [col for col in df.columns if "Flat" in col]
    df = (
        df.with_columns(
            pl.sum_horizontal(flats_cols).alias("Flat, maisonette or apartment")
        )
        .drop(flats_cols)
        .rename(
            {
                col: col.replace("Whole house or bungalow: ", "")
                for col in df.select(cs.numeric()).columns
            }
        )
        .rename({"Type of accomodation": "lsoa"})
    )

    return df


def transform_df_target_tenure() -> pl.DataFrame:
    """
    Load and transform tenure type counts per LSOA/data zone for England, Scotland, and Wales from census data.

    Returns:
        pl.DataFrame: tenure type counts per LSOA/data zone for England, Scotland, and Wales
    """
    ew_df = load_transform_df_target_tenure_ew()
    s_df = load_transform_df_target_tenure_scotland()
    s_df = s_df.select(ew_df.columns)

    return pl.concat([ew_df, s_df], how="vertical")


def load_transform_df_target_tenure_scotland() -> pl.DataFrame:
    """
    Load and transform tenure type counts per data zone in Scotland from census data.

    Returns:
        pl.DataFrame: tenure type counts per data zone in Scotland
    """
    df = pl.read_csv(
        config["data_source"]["S_census_tenure"],
        skip_rows=10,
        columns=list(range(1, 4)),
        infer_schema_length=10000,
    )
    df = (
        df.drop_nulls()
        .rename({"Intermediate Zone - Data Zone 2011": "lsoa"})
        .pivot("Household Tenure", index="lsoa", values="Count")
        .drop([col for col in df.columns if "Total" in col])
    )
    private_rental = [col for col in df.columns if "Private" in col]
    private_rental.extend(["Lives Rent Free"])
    df = df.with_columns(
        pl.sum_horizontal([col for col in df.columns if "Owned" in col]).alias(
            "owner_occupied"
        ),
        pl.sum_horizontal(private_rental).alias("private_rental"),
        pl.sum_horizontal([col for col in df.columns if "Social" in col]).alias(
            "social_rental"
        ),
    )
    return df.select(["lsoa", "owner_occupied", "social_rental", "private_rental"])


def load_transform_df_target_tenure_ew() -> pl.DataFrame:
    """
    Get dataframe of tenure type counts for all LSOAs in England and Wales from census data.

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
                pl.sum_horizontal(owned_cols).alias("owner_occupied"),
                pl.sum_horizontal(social_rent_cols).alias("social_rental"),
                pl.sum_horizontal(private_rent_cols).alias("private_rental"),
            ]
        )
        .select(pl.col(["lsoa", "owner_occupied", "social_rental", "private_rental"]))
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


def get_df_target_build_year_la() -> pl.DataFrame:
    """
    Get dataframe of counts of properties built before and after 1930 for local authorities in England and Wales.
    Source: Consumer Data Research Centre, 2015.

    Returns:
        pl.Dataframe: counts of properties built before and after 1930 for local authorities in England and Wales.
    """
    df = pl.read_parquet(config["data_source"]["EW_cdrc_dwelling_age_la"])
    df = df.select(["lsoa", "pre_1930", "post_1930", "unknown"])

    return df
