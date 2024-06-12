import polars as pl
import polars.selectors as cs
from asf_heat_pump_suitability.getters import get_datasets


def generate_df_garden_space_avg() -> pl.DataFrame:
    """
    Generate dataframe with mean average garden size (m2) by property type (house; flat; or unknown) for each MSOA.

    Returns:
        pl.DataFrame: mean average garden size (m2) by property type for each MSOA
    """
    df = get_datasets.get_df_ons_garden_space_avg()
    df = clean_df_garden_space_column_names(df)

    for property_type in ["Houses", "Flats", "Total"]:
        df = _calculate_cols_avg_garden_space(df, property_type=property_type)

    df = (
        df.select(
            [
                "MSOA code",
                "Houses",
                "Flats",
                "Total",  # This is the overall average for houses & flats. This will be joined to properties of unknown type
            ]
        )
        .rename({"Total": "unknown"})
        .melt(id_vars="MSOA code", value_vars=cs.numeric())
        .rename(
            {
                "variable": "msoa_avg_outdoor_space_property_type",
                "value": "msoa_avg_outdoor_space_m2",
            }
        )
    )

    return df


def clean_df_garden_space_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create unique descriptive column names for ONS garden space dataset. Column names are split across first two rows
    in the raw dataset.

    Args:
        df (pl.DataFrame): ONS garden space dataset

    Returns:
        pl.DataFrame: ONS garden space dataset with clean column names
    """
    # Get column suffixes from the first row of data
    suffixes = ["" if not v else v for v in df.row(0)]

    # For unnamed columns, rename column with the name of the previous column
    cols = list(df.columns)
    for i, col in enumerate(cols):
        if "UNNAMED" in col:
            cols[i] = cols[i - 1]

    # Join suffixes onto column names where applicable
    df.columns = [
        " ".join([col, suffix]).strip() for col, suffix in zip(cols, suffixes)
    ]
    df = df[1:]  # remove first row of dataset containing column suffixes

    return df


def _calculate_cols_avg_garden_space(
    df: pl.DataFrame, property_type: str
) -> pl.DataFrame:
    """
    Calculate mean average garden size (m2) for all properties (including those without gardens) in each property type.
    Raw ONS gardens dataset only contains mean average garden size for properties with gardens.

    Args:
        df (pl.DataFrame): ONS garden space dataset
        property_type (str): name of property type. Options: "Houses", "Flats", "Total"

    Returns:
        pl.DataFrame: ONS garden space dataset with mean average garden size (m2) for all properties
    """
    df = df.with_columns(
        (
            pl.col(
                f"Property type: {property_type} Private outdoor space total area (m2)"
            ).cast(pl.Float64)
            / pl.col(f"Property type: {property_type} Address count").cast(pl.Float64)
        ).alias(f"{property_type}")
    )
    return df


def _correct_col_recalculate_avg_garden_space_flats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recalculate average garden space for flats in ONS garden space dataset because column appears to be erroneously
    misaligned.

    Args:
        df (pl.DataFrame): ONS garden space dataset

    Returns:
        pl.DataFrame: ONS garden space dataset with corrected average garden space for flats
    """
    df = df.with_columns(
        pl.col("Property type: Flats Private outdoor space total area (m2)")
        / pl.col("Property type: Flats Adress with private outdoor space count").alias(
            "Property type: Flats Average size of private outdoor space (m2)"
        )
    )

    return df
