import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def prepare_df_land_area_ons() -> pl.DataFrame:
    """
    Process and clean ONS land area dataset

    Args

    Returns
        pl.DataFrame: processed ONS land area
    """
    df = get_datasets.get_df_ons_land_area()
    df = preprocess_df_land_area(df)
    return df


def preprocess_df_land_area(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess

    Args
        df (pl.DataFrame): dataset of land areas
        pcd_col (str): name of column containing land areas

    Returns
        pl.DataFrame: dataset with renamed LSOA column
    """
    # Rename columns
    df = df.rename(
        {
            "LSOA21CD": "lsoa21",
        }
    )

    return df
