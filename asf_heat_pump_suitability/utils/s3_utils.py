"""
This file contains utility functions for interacting with Amazon S3.
"""

import boto3
from typing import List


def fetch_list_file_paths_from_s3_folder(
    s3_client: boto3.client, s3_bucket: str, path_folder: str, file_type: str = None
) -> List[str]:
    """
    Fetches file paths from a specified S3 folder.

    Args:
        s3_client (boto3.client): An initialized boto3 S3 client.
        s3_bucket (str): The name of the S3 bucket.
        path_folder (str): The path to the folder in S3.
        file_type (str): The type of files to fetch (e.g., ".parquet", ".csv", ".geojson"). If None, fetches all files.

    Returns:
        List[str]: list of strings with the file paths.
    """
    # Normalize prefix: ensuring it ends with '/'
    if path_folder and not path_folder.endswith("/"):
        path_folder += "/"

    # Set a paginator to handle large number of files (otherwise only first 1000 files are returned)
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=path_folder)

    file_paths = []

    # Iterate through each page of results and extract file paths
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Filter to include only files of the specified type (if file_type is provided)
            if file_type is None or key.endswith(file_type):
                file_paths.append(key)

    return file_paths


def extract_tuple_bucket_prefix(s3_uri: str) -> tuple:
    """
    Extract bucket name and prefix (folder key) from an S3 URI.

    Args:
        s3_uri (str): S3 URI

    Returns:
        tuple: bucket name, folder prefix
    """
    bucket_name = s3_uri.split("s3://")[1].split("/")[0]
    prefix = s3_uri.split(f"s3://{bucket_name}/")[1]

    return (bucket_name, prefix)
