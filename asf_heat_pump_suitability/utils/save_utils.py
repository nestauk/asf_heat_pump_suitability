import s3fs
import polars as pl


def save_parquet_to_s3(df: pl.DataFrame, path: str) -> None:
    """
    Save dataframe as parquet file to S3.

    Args:
        df (pl.DataFrame): dataframe
        path (str): path to S3 destination

    Returns:
        None
    """
    fs = s3fs.S3FileSystem()
    with fs.open(path=path, mode="wb") as f:
        df.write_parquet(f)
