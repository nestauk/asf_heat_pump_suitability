"""
Run full pipeline end-to-end for any number of Local Authorities in batches using `orbit`.

Splits the given local authorities into chunks and launches one `orbit` job per chunk, each running `pipeline.py`
remotely for that chunk's local authorities.

Usage:
    python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py --local_authorities <LOCAL_AUTHORITY> [<LOCAL_AUTHORITY> ...] [--size N [--is_number_of_chunks]] --model_release YYYYMMDD [--release_date YYYYMMDD] [--compare [--trigger TRIGGER] [--old_release_date YYYYMMDD]]

    e.g. to run for Great Britain in chunks of 20 local authorities each:
        python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py --local_authorities GB --size 20

    e.g. to run for two named local authorities as a single chunk:
        python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py --local_authorities 'glasgow city' 'south lanarkshire'

    e.g. to run for all Local Authorities Great Britain in 10 chunks:
        python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py --local_authorities GB --size 10 --is_number_of_chunks

Pass `--local_authorities GB` to run for every local authority in GB, otherwise pass one or more local authority names.
`--size` sets the chunk size by default (the number of Local Authorities to be run in each batch), or the number of
chunks if `--is_number_of_chunks` is also passed (the number of batches to run in parallel). If `--size` is
omitted, all local authorities are run in a single chunk.

Pass `--prod` to push changes to production (i.e. staging area). This should only be used when running from `dev`.
With `--prod`, the launcher reads `front_end_s3_bucket` and `front_end_staging_s3_path` from `.env` and passes them
to the jobs.

The release date defaults to today and is pinned across all stages, so a run crossing midnight still writes to a single
dated release directory.

`--compare`, `--trigger` and `--old_release_date` are passed to every chunk's `pipeline.py` run, which then compares
each local authority's stage outputs with their latest earlier versions, or with `--old_release_date` when given (see
`pipeline.py`).
"""

import math
import numpy as np
import argparse
from typing import List

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.run.pipeline import (
    add_compare_arguments,
    check_compare_arguments,
)
from asf_heat_pump_suitability.utils import save_utils


def chunk_list_strings(arr: np.array, n: int, size: bool) -> List[List[str]]:
    """
    Split an array of strings into chunks for processing.

    Args:
        arr (np.array): array of strings to chunk.
        n (int): size of chunks if `size` set to `True` otherwise, number of chunks.
        size (bool): set to `True` if `n` represents size of chunks, or `False` if `n` represents number of chunks.

    Returns:
        List[List[str]]: list of lists of strings
    """
    if not size:
        # Calculate how many items per chunk
        n = math.ceil(len(arr) / n)
    chunks = [list(l) for l in np.array_split(arr, n)]
    print(f"{len(chunks)} chunks to process. Each approx. {n} items long.")

    return chunks


def validate_str_cli_arguments(args: List[str]) -> List[str]:
    """
    Add a '--' prefix to each argument that does not already have one.

    Args:
        args (List[str]): arguments to pass to `orbit`.

    Returns:
        List[str]: arguments prefixed with '--'
    """
    return [a if a.startswith("--") else "--" + a for a in args]


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        required=True,
        default="GB",
    )

    parser.add_argument(
        "--size",
        help="Number of chunks for batch processing if `--number_of_chunks` passed, otherwise size of chunks for batch processing.",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--is_number_of_chunks",
        help="Pass this argument if `--size` represents number of chunks.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--model_release",
        help="Date of model training in YYYYMMDD format.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    parser.add_argument(
        "--prod",
        help="Set to push changes to production (i.e. staging area). This should only be used when running from `dev`.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-o",
        "--orbit_args",
        help="Optional command line arguments to pass to `orbit`, e.g. '--on-demand'.",
        required=False,
        type=str,
        nargs="+",
    )

    # The comparison flags are checked here, so a bad flag
    # stops before any Orbit job starts.
    add_compare_arguments(parser=parser)

    args = parser.parse_args()
    check_compare_arguments(parser=parser, args=args)
    return args


if __name__ == "__main__":
    import os
    import sys
    import polars as pl
    from tqdm import tqdm
    import subprocess
    from dotenv import load_dotenv
    from asf_heat_pump_suitability.utils import s3_utils
    from asf_heat_pump_suitability.pipeline.transform import local_authority

    args = parse_arguments()
    additional_flags = ["--prod"] if args.prod else []
    if args.orbit_args:
        additional_flags.extend(validate_str_cli_arguments(args.orbit_args))

    if args.prod:
        # .env is not uploaded to orbit, so pass the front-end S3 location to the jobs explicitly
        load_dotenv()
        front_end_vars = ("front_end_s3_bucket", "front_end_staging_s3_path")
        missing = [name for name in front_end_vars if not os.environ.get(name)]
        if missing:
            sys.exit(f"--prod needs {', '.join(missing)} in your .env file.")
        for name in front_end_vars:
            additional_flags += ["-e", f"{name}={os.environ[name]}"]
    local_authorities = args.local_authorities
    release_date = save_utils.get_str_release_date(args.release_date)
    print(f"Release date pinned to: {release_date}")

    if local_authorities == ["GB"]:
        local_authorities = local_authority.load_list_la_names(nation="GB")

    chunks = chunk_list_strings(
        np.asarray(local_authorities), n=args.size, size=args.is_number_of_chunks
    )

    compare_flags = ["--compare"] if args.compare else []
    if args.trigger:
        compare_flags += ["--trigger", args.trigger]
    if args.old_release_date:
        compare_flags += ["--old_release_date", args.old_release_date]

    for chunk in tqdm(chunks):
        print(f"Running chunk: {chunk}")
        subprocess.run(
            [
                "orbit",
                "launch",
                "--script",
                "asf_heat_pump_suitability/pipeline/run/pipeline.py",
                "--team",
                "asf",
                "--project",
                "local_heat_planning_tool",
                "--cpu",
                "8",
                "--memory",
                "60gb",
                "-e",
                "PYTHONPATH=/app",  # Install repository in docker image
                "--local_authorities",
                *chunk,
                "--model_release",
                args.model_release,
                "--release_date",
                release_date,
                *additional_flags,
                *compare_flags,
            ]
        )

    # Aggregate any error logs and save to a single file
    dir_uri = (
        config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date, local_authorities="REMOVE"
        )
    ).rsplit("/", 1)[0]
    failure_paths = s3_utils.list_files_s3_path(dir_uri=str(dir_uri))
    if failure_paths:
        failures = pl.concat([pl.read_csv(f) for f in failure_paths])
        save_as = config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date, local_authorities="ORBIT_CONCAT"
        )
        failures.write_csv(save_as)
        if failures.filter(pl.col("level") == "error").height > 0:
            print(f"Pipeline failed for some local authorities. See logs at {save_as}")
        else:
            print(
                f"Pipeline finished for all local authorities, some with warnings. See logs at {save_as}"
            )
    else:
        print("Pipeline finished for all Local Authorities successfully!")
