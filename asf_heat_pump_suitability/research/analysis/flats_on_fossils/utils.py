import polars as pl
import polars.selectors as cs

from asf_heat_pump_suitability import config


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
    df = pl.scan_csv(config["data_source"]["EW_census_accommodation_type"]).collect()
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
                    "Part of a converted or shared house, including bedsits",
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
    df = (
        pl.scan_csv(
            config["data_source"]["S_census_accommodation_type"],
            skip_rows=10,
            infer_schema_length=10000,
        )
        .collect()
        .select(pl.nth(range(0, 11)))
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
        .rename(
            {"Type of accomodation": "lsoa"}
        )  # The Data Zone (lsoa) column name is mislabelled due to .csv formatting
    )

    # A small number of rows seem to erroneously have zero values for all property types, we need to remove them
    df = df.filter(
        pl.sum_horizontal(
            [
                "Detached",
                "Semi-detached",
                "Terraced (including end-terrace)",
                "Caravan or other mobile or temporary structure",
                "Flat, maisonette or apartment",
            ]
        )
        != 0
    )

    return df
