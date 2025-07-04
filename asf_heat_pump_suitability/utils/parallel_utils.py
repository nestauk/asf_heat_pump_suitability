from typing import List
import polars as pl


def chunk_df_by_group(df: pl.DataFrame, group_col: str, n: int) -> List[pl.DataFrame]:
    """
    Split dataframe into chunks of group IDs.

    Args:
        df (pl.DataFrame): dataframe
        group_col (str): column with group IDs
        n (int): number of group IDs in each chunk

    Returns:
        List[pl.DataFrame]: list of dataframe chunks
    """
    # Get unique group IDs
    group_ids = list(df[group_col].drop_nulls().unique())

    # Get groups of IDs
    group_chunks = [group_ids[i : i + n] for i in range(0, len(group_ids), n)]

    return [df.filter(pl.col(group_col).is_in(chunk)) for chunk in group_chunks]
