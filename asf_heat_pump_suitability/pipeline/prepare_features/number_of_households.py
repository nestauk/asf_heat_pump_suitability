import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def prepare_df_num_of_households_ons() -> pl.DataFrame:
    """
    Process and clean ONS number of households dataset

    Args

    Returns
        pl.DataFrame: processed ONS number of households
    """
    df = get_datasets.get_df_ons_number_of_households()
    df = preprocess_df_num_of_households(df)
    return df


def preprocess_df_num_of_households(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess

    Args
        df (pl.DataFrame): dataset containing number of households
        pcd_col (str): name of column containing postcodes

    Returns
        pl.DataFrame: dataset with standardised postcode column
    """
    # Rename columns
    df = df.rename(
        {
            "mnemonic": "lsoa21",
            "2021": "Number of households 2021",
        }
    )

    return df


prepare_df_num_of_households_ons()
