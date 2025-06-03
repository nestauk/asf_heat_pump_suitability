from typing import List
import pandas as pd


def chunk_df(df: pd.DataFrame, size: int) -> List[pd.DataFrame]:
    """
    Split dataframe into chunks of specified size.

    Args:
        df (pl.DataFrame): dataframe
        size (int): number of records per chunk

    Returns:
        List[pl.DataFrame]: list of dataframe chunks
    """
    return [df.iloc[i : i + size] for i in range(0, len(df), size)]
