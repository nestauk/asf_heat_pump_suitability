import s3fs
import polars as pl
import logging
import boto3


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


def upload_file_to_s3(
    local_file_path: str,
    s3_bucket: str,
    s3_key_dir: str,
    save_to_s3: bool,
    filename: str,
    subfolder: str,
):
    """
    Upload a local file to an S3 bucket, used for evaluation of our heat network suitability model using the DESNZ Heat Network pilot zones.

    Args:
        local_file_path (str): Path to the local file.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): S3 key (path) where the file should be uploaded.
        save_to_s3 (bool): boolean which indicates whether to save or not the file to s3.
        filename (str): The actual filename to store in S3.
        subfolder (str): Subfolder within S3.
    """
    if save_to_s3:
        s3_client = boto3.client("s3")
        s3_key = f"{s3_key_dir}{subfolder}/{filename}"
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        logging.info(f"File uploaded to s3://{s3_bucket}/{s3_key}")
