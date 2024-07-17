# Add property density feature to EPC dataset
import polars as pl


def extend_df_with_property_density(enhanced_epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add property density feature to the EPC dataset
    Args:
        enhanced_epc_df (pl.DataFrame): EPC dataset with additional features
    Returns:
        pl.DataFrame: EPC dataset with property density feature
    """
    # Fill None values with a default value (e.g., 0)
    enhanced_epc_df = enhanced_epc_df.with_columns(
        [
            pl.col("Number of households 2021")
            .fill_null(0)
            .alias("Number of households 2021"),
            pl.col("Land Count (Area in KM2)")
            .fill_null(0)
            .alias("Land Count (Area in KM2)"),  # Fill with 1 to avoid division by zero
        ]
    )
    # Calculate and add the new column with the property density, handling division by zero
    enhanced_epc_df = enhanced_epc_df.with_columns(
        (
            pl.when(pl.col("Land Count (Area in KM2)") != 0)
            .then(
                pl.col("Number of households 2021") / pl.col("Land Count (Area in KM2)")
            )
            .otherwise(None)
        ).alias("Property density (households per KM2)")
    )
    # Changes 0 values back to None
    enhanced_epc_df = replace_zeros_with_none_df(
        enhanced_epc_df, "Number of households 2021"
    )
    enhanced_epc_df = replace_zeros_with_none_df(
        enhanced_epc_df, "Land Count (Area in KM2)"
    )

    return enhanced_epc_df


def replace_zeros_with_none_df(df: pl.Dataframe, column_name: str) -> pl.DataFrame:
    """
    This function replaces 0 values in the specified column of a DataFrame with None.

    Parameters:
    df (pl.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column in which to replace 0 values.

    Returns:
    pl.DataFrame: The modified DataFrame.
    """
    df = df.with_columns(
        pl.when(pl.col(column_name) == 0)
        .then(None)
        .otherwise(pl.col(column_name))
        .alias(column_name)
    )

    return df
