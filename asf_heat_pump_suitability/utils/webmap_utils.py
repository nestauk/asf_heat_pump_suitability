import os
from typing import Iterator
from urllib.parse import urlparse

import boto3
import botocore
import fsspec
import pandas as pd
import s3fs


def _generate_str_s3_object_keys(
    s3_paginator: botocore.paginate.Paginator, bucket_name: str, prefix: str = "/"
) -> Iterator[str]:
    """S3 bucket key generator.

    Necessary for getting all available keys when there are >1000 objects in the bucket.

    Args:
        s3_paginator (botocore.paginate.Paginator): boto3/botocore paginator object.
        bucket_name (str): s3 bucket name.
        prefix (str): filter the paginated results by prefix, default '/'.

    Yields:
        str: object keys from s3 bucket.
    """
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for content in page.get("Contents", ()):
            yield content["Key"]


def get_str_latest_scores_parquet_file_uri() -> str:
    """Get uri of latest heat pump suitability scores parquet file in S3.

    Returns:
        str: uri of latest heat pump suitability scores parquet file

    Raises:
        FileNotFoundError: if latest heat pump suitability scores parquet file not found in S3 bucket.
    """
    # Make s3 client
    s3_paginator = boto3.client("s3").get_paginator("list_objects_v2")
    # Get candidates
    candidates = [
        key
        for key in _generate_str_s3_object_keys(s3_paginator, "asf-heat-pump-suitability", prefix="outputs/")
        if "/suitability/" in key
    ]
    # Get unique Year-Quarters, which conceptually represent folders.
    year_quarters = pd.to_datetime(
        list(set([candidate.split("/")[1].replace("Q", "-Q") for candidate in candidates]))
    ).sort_values(ascending=False)
    # Iterate over year_quarter to find the most recent heat_pump_suitability_per_lsoa.parquet
    candidate_file = None
    i = 0
    try:
        while not candidate_file:
            # Iterate as long as you haven't identified a candidate file.
            year_quarter = year_quarters[i]
            year_quarter_str = year_quarter.to_period("Q").strftime("%YQ%q")
            # get candidates from the required year_quarter, filename and date structure.
            # Assumes that 8 characters followed by _ at start of file is a date.
            year_quarter_candidates = [
                candidate
                for candidate in candidates
                if (f"/{year_quarter_str}/" in candidate)
                & ("heat_pump_suitability_per_lsoa.parquet" in candidate)
                & (candidate.split("/")[-1].split("_")[0].__len__() == 8)
            ]
            if len(year_quarter_candidates) == 1:
                # if only 1 option, use that.
                candidate_file = year_quarter_candidates[0]
            elif len(year_quarter_candidates) > 1:
                # get most recent dated file
                year_quarter_candidates_dates = [
                    candidate.split("/")[-1].split("_")[0] for candidate in year_quarter_candidates
                ]
                # argmax will return the first max index if there are multiple matches.
                latest_file_id = pd.to_datetime(year_quarter_candidates_dates).argmax()
                # use the most recently dated file
                candidate_file = year_quarter_candidates[latest_file_id]
            else:
                # increment
                i += 1
    except Exception as e:
        # If iteration fails it will likely be due to an index error on year_quarter.
        # However the root cause is file not found, so raise that error.
        raise FileNotFoundError(
            "Could not find latest suitability score file automatically, please enter filepath manually."
        ) from e

    return f"s3://asf-heat-pump-suitability/{candidate_file}"


def check_exists_str_file_uri(filestring: str) -> str:
    """Check if filestring passed exists and return.

    Args:
        filestring (str): location of file to check.

    Returns:
        str: filestring if file exists.

    Raises:
        FileNotFoundError: if file does not exist.
    """
    # First check if local file
    fs = fsspec.filesystem("file")
    if fs.exists(filestring):
        return filestring
    # Now check if it's an s3 file
    fs = s3fs.S3FileSystem()
    if fs.exists(filestring):
        return filestring
    # If it's not a local or s3 file, raise an error.
    raise FileNotFoundError(f"Couldn't find {filestring} as either a local or s3-based file.")


def check_exists_str_output_directory(directorystring: str) -> str:
    """Check if the output directory exists.

    Args:
        directorystring (str): location of directory to check.

    Returns:
        str: directorystring if directory exists.

    Raises:
        OSError: if directory does not exist.
    """
    uri = urlparse(directorystring)
    # if s3, test bucket exists
    if uri.scheme == "s3":
        s3 = boto3.resource("s3")
        try:
            s3.meta.client.head_bucket(Bucket=uri.netloc)
            return directorystring
        except Exception as e:
            raise OSError(f"Couldn't connect to S3 Bucket: {uri.netloc}, check it exists and is accessible.") from e
    elif uri.scheme == "":
        # assume local file, test if it exists
        if os.path.isdir(directorystring):
            return directorystring
    raise OSError(f"Couldn't connect to {directorystring}, check it exists and is accessible.")
