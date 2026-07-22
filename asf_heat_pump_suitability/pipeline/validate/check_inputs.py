"""
Preflight check that all S3 input paths configured under `config["data"]` exist.

Recursively collects every s3:// path in the nested `config["data"]` mapping,
truncates templated paths (e.g. ".../{square}/{square}_{layer}.shp") to the
prefix before the first "{" token, and checks each one exists in S3. All
missing paths are reported in one pass and the script exits non-zero if any
are missing, so run_pipeline.sh can abort before the local-authority loop
starts. Templated paths are checked at prefix level only, not resolved per
grid square or local authority.

Usage:
    python asf_heat_pump_suitability/pipeline/validate/check_inputs.py
"""

import logging
import sys

import boto3

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import s3_utils


def get_list_s3_paths(config_section: dict) -> list:
    """
    Recursively collect s3:// path leaves from a nested config mapping.

    Args:
        config_section: nested mapping of config values, e.g. `config["data"]`.

    Returns:
        list: all string leaves starting with "s3://", in traversal order.
    """
    paths = []
    for value in config_section.values():
        if isinstance(value, dict):
            paths.extend(get_list_s3_paths(value))
        elif isinstance(value, str) and value.startswith("s3://"):
            paths.append(value)
    return paths


def get_str_checkable_prefix(path: str) -> str:
    """
    Truncate a templated S3 path to the prefix before its first "{" token.

    Args:
        path: S3 path, optionally containing template tokens such as "{square}".

    Returns:
        str: the path unchanged, or its prefix before the first "{".
    """
    return path.split("{", 1)[0]


def get_list_missing_s3_paths(paths: list, s3_client: boto3.client) -> list:
    """
    Check each configured S3 path exists and collect the missing ones.

    Args:
        paths: configured s3:// paths, possibly templated.
        s3_client: an initialized boto3 S3 client.

    Returns:
        list: configured paths with no S3 object at or under their
            (truncated) prefix.
    """
    missing_paths = []
    for path in paths:
        bucket, _, key = (
            get_str_checkable_prefix(path).removeprefix("s3://").partition("/")
        )
        if not s3_utils.get_bool_s3_path_exists(s3_client, bucket, key):
            missing_paths.append(path)
    return missing_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_paths = get_list_s3_paths(config["data"])
    missing_input_paths = get_list_missing_s3_paths(input_paths, boto3.client("s3"))
    for missing_path in missing_input_paths:
        logging.error(f"Missing S3 input path: {missing_path}")
    if missing_input_paths:
        logging.error(
            f"{len(missing_input_paths)} of {len(input_paths)} configured S3 input "
            "paths missing."
        )
        sys.exit(1)
    logging.info(f"All {len(input_paths)} configured S3 input paths exist.")
