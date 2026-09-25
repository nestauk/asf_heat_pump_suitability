"""
Preflight check that all S3 input paths configured under `config["data"]` exist.

Recursively collects every s3:// path in the nested `config["data"]` mapping,
expands paths with a "{square}" token into one path per grid square, truncates
any remaining tokens (e.g. "{layer}") to the prefix before the first "{", and
checks each resulting path exists in S3. All missing paths are reported in
one pass and the script exits non-zero if any are missing, so
run_pipeline.sh can abort before the local-authority loop starts. A missing
grid square is reported individually; layer files are only required
at-least-one-per-square, since some layers are legitimately absent in a
square.

Grid squares are derived the same way the pipeline derives them
(`local_authority.get_list_la_grid_squares`): the BNG grid clipped to the
named local authorities' boundaries, or to all of GB when no local
authorities are given, so sea-only squares with no OS data are never checked.
Square/product combinations listed in
`config["constant"]["os_data_absent_grid_squares"]` (remote islands and
NI-overlap squares specific OS products ship no files for) are also skipped,
per product.

Corollary: every path under `config["data"]` is treated as a required
production input, except research-only paths listed in `RESEARCH_ONLY_PATHS`,
which are skipped until production and research configs are split.

Usage:
    python asf_heat_pump_suitability/pipeline/validate/check_inputs.py \
        [--local_authorities <names>]

Defaults to checking the whole of GB when --local_authorities is omitted.
"""

import argparse
import logging
import sys

import boto3

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.transform import local_authority
from asf_heat_pump_suitability.utils import s3_utils

# TODO(#469): remove RESEARCH_ONLY_PATHS once production and research configs
# are split. These are inputs configured under config["data"] but only read by
# research scripts, so the preflight skips them.
RESEARCH_ONLY_PATHS = {
    "s3://asf-local-heat-planning-tool/inputs/geodata/council_tax/PLYMOUTH_CTBANDS_ONSUD_202512.csv",
}


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


def get_set_squares_absent_for_path(path: str, absent_squares: dict) -> set:
    """
    Collect the grid squares that publish no data for the product in a path.

    A path belongs to a product when the product's folder name appears in it,
    e.g. "oproad_essh_gb" in ".../oproad_essh_gb/20260708/{square}_RoadLink.shp".

    Args:
        path: configured s3:// path, templated or not.
        absent_squares: mapping of product folder name to the squares that
            product publishes no data for.

    Returns:
        set: squares to skip for this path; empty if no product matches.
    """
    squares_to_skip = set()
    for product_folder, product_squares in absent_squares.items():
        if product_folder in path:
            squares_to_skip.update(product_squares)
    return squares_to_skip


def generate_list_expanded_square_paths(
    paths: list, squares: list, absent_squares: dict = None
) -> list:
    """
    Expand each "{square}"-templated path into one path per grid square.

    Args:
        paths: configured s3:// paths, possibly templated.
        squares: 100km OS grid square codes, e.g. ["SX", "SD"].
        absent_squares: optional mapping of product folder name (matched as a
            substring of the path) to squares that product publishes no data
            for; those square expansions are skipped for that path.

    Returns:
        list: paths with "{square}" substituted per square; paths without the
            token are passed through unchanged.
    """
    absent_squares = absent_squares or {}
    expanded_paths = []
    for path in paths:
        if "{square}" not in path:
            expanded_paths.append(path)
            continue
        squares_to_skip = get_set_squares_absent_for_path(path, absent_squares)
        expanded_paths.extend(
            path.replace("{square}", square)
            for square in squares
            if square not in squares_to_skip
        )
    return expanded_paths


def get_str_common_prefix(path: str) -> str:
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
        bucket, sep, key = (
            get_str_common_prefix(path).removeprefix("s3://").partition("/")
        )
        if not s3_utils.get_bool_s3_path_exists(s3_client, bucket, key):
            missing_paths.append(path)
    return missing_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire. Defaults to the whole of GB.",
        type=str,
        nargs="+",
        default=["GB"],
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    resolved_las = local_authority.resolve_list_la_names(args.local_authorities)
    grid_squares = sorted(local_authority.get_list_la_grid_squares(resolved_las))
    logging.info(
        f"Checking inputs for {'whole of GB' if resolved_las is None else ', '.join(resolved_las)} "
        f"({len(grid_squares)} grid squares)."
    )
    input_paths = [
        path
        for path in get_list_s3_paths(config["data"])
        if path not in RESEARCH_ONLY_PATHS
    ]
    if not input_paths:
        logging.error(
            "No S3 input paths found under config['data'] - check config parsing."
        )
        sys.exit(1)
    input_paths = generate_list_expanded_square_paths(
        input_paths,
        grid_squares,
        absent_squares=config["constant"]["os_data_absent_grid_squares"],
    )
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
