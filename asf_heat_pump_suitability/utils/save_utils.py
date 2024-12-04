import s3fs
import polars as pl
import logging


def save_to_s3(df: pl.DataFrame, path: str) -> None:
    """
    Save dataframe as parquet file to S3.

    Args:
        df (pl.DataFrame): dataframe
        path (str): path to S3 destination

    Returns:
        None
    """
    logging.info(f"Saving file to {path}")
    file_type = path.split(".")[-1]
    fs = s3fs.S3FileSystem()
    if file_type == "parquet":
        with fs.open(path=path, mode="wb") as f:
            df.write_parquet(f)
    elif file_type == "csv":
        with fs.open(path=path, mode="wb") as f:
            df.write_csv(f)
    else:
        raise ValueError(
            "Save to S3 can only save .parquet or .csv file types."
            "Please ensure the `path` argument contains one of these file types."
        )
