"""
This file contains utility functions for interacting with Amazon S3.
"""

import boto3
from typing import List


def fetch_list_file_paths_from_s3_folder(
    s3_client: boto3.client,
    s3_bucket: str,
    path_folder: str,
    file_type: str | List[str] = None,
) -> List[str]:
    """
        Fetches file paths from a specified S3 folder.
    `
        Args:
            s3_client (boto3.client): An initialized boto3 S3 client.
            s3_bucket (str): The name of the S3 bucket.
            path_folder (str): The path to the folder in S3.
            file_type (str | List[str]): The type of files to fetch (e.g., ".parquet", ".csv", ".geojson"). If None, fetches all files.

        Returns:
            List[str]: list of strings with the file paths.
    """
    # Normalize prefix: ensuring it ends with '/'
    if path_folder and not path_folder.endswith("/"):
        path_folder += "/"

    # Normalize file_type to a tuple for str.endswith()
    if isinstance(file_type, str):
        file_types = (file_type,)
    elif isinstance(file_type, list):
        file_types = tuple(file_type)
    else:
        file_types = None

    # Set a paginator to handle large number of files (otherwise only first 1000 files are returned)
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=path_folder)

    file_paths = []

    # Iterate through each page of results and extract file paths
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Skip directory markers (e.g., if path_folder itself is returned)
            if key.endswith("/"):
                continue

            # Filter to include only files of the specified type (if file_type is provided)
            if file_types is None or key.lower().endswith(file_types):
                file_paths.append(key)

    return file_paths
