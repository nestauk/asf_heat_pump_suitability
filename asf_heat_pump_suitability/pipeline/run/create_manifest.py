"""
Creates a manifest.json file with metadata about all local authority datasets.
"""

from dotenv import load_dotenv
import boto3
from typing import List
import os
import json
from smart_open import open


def fetch_file_paths_from_s3_folder(
    s3_bucket: str, path_folder: str, file_type: str = None
) -> List[str]:
    """
    Fetches file paths from a specified S3 folder.
    Args:
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


if __name__ == "__main__":
    from asf_heat_pump_suitability.utils import save_utils

    # Load environment variables from .env file
    load_dotenv()

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Load file names in s3 path
    front_end_s3_bucket = os.environ.get("front_end_s3_bucket")
    front_end_staging_s3_path = os.environ.get("front_end_staging_s3_path")

    geojson_file_paths_list = fetch_file_paths_from_s3_folder(
        s3_bucket=front_end_s3_bucket,
        path_folder=front_end_staging_s3_path,
        file_type=".geojson",
    )

    geojson_file_names_list = [
        file_path.split("/")[-1] for file_path in geojson_file_paths_list
    ]

    # create a dictionary with "id", "local_authority", and "geojson_url" for each local authority, where the "geojson_url" is the url to the geojson file in the front-end S3 bucket
    geojson_s3_url_prefix = os.environ.get("geojson_s3_url_prefix")

    geojson_dict_list = []

    for file_name, file_path in zip(geojson_file_names_list, geojson_file_paths_list):
        with open(os.path.join("s3://", file_path), "r") as f:
            geojson_data = json.load(f)

        # Extract metadata
        local_authority = geojson_data.get("metadata").get("Local authority")

        geojson_dict_list.append(
            {
                "id": local_authority,
                "local_authority": local_authority.replace("_", " ").title(),
                "geojson_url": os.path.join(geojson_s3_url_prefix, file_name),
            }
        )

    # Save the list of dictionaries as a manifest.json file in the front-end S3 bucket
    save_utils.save_to_s3(
        geojson_dict_list,
        os.path.join(front_end_staging_s3_path, "manifest.json"),
    )
