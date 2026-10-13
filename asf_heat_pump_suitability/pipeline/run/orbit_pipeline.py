"""
Run full pipeline end-to-end for any number of Local Authorities in batches using `orbit`.

Splits the given local authorities into chunks and launches one `orbit` job per chunk, each running `pipeline.py`
remotely for that chunk's local authorities.

Usage:
    python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py --local_authorities <LOCAL_AUTHORITY> [<LOCAL_AUTHORITY> ...] [--size N [--is_number_of_chunks]] [--release_date YYYYMMDD] [--compare [--trigger TRIGGER] [--old_release_date YYYYMMDD]]

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
        "--compare",
        help="Pass `--compare` to every chunk, to compare each stage's output with its latest earlier version.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--trigger",
        help="Why the run is made; passed to every chunk's comparisons to check their tolerances.",
        choices=sorted(config["compare_versions"]["tolerances"]),
        default=None,
    )

    parser.add_argument(
        "--old_release_date",
        help="Passed to every chunk: compare every stage against this release instead of each stage's latest earlier one.",
        type=save_utils.get_str_release_date,
        default=None,
    )

    args = parser.parse_args()
    if args.trigger is not None and not args.compare:
        parser.error("--trigger is used only by the comparisons; add --compare.")
    if args.old_release_date is not None:
        if not args.compare:
            parser.error(
                "--old_release_date is used only by the comparisons; add --compare."
            )
        try:
            release_date = save_utils.get_str_release_date(args.release_date)
        except ValueError as error:
            parser.error(str(error))
        if args.old_release_date >= release_date:
            parser.error("--old_release_date must be earlier than the release date.")
    return args


if __name__ == "__main__":
    from pathlib import Path
    import polars as pl
    from tqdm import tqdm
    import subprocess
    from asf_heat_pump_suitability.utils import s3_utils

    args = parse_arguments()
    local_authorities = args.local_authorities
    release_date = save_utils.get_str_release_date(args.release_date)
    print(f"Release date pinned to: {release_date}")

    if local_authorities == ["GB"]:
        local_authorities = pl.read_csv(config["data"]["processed"]["valid_la_names"])[
            "LAD23NM"
        ].to_list()

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
        if args.prod:
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
                    "--release_date",
                    release_date,
                    "--prod",
                    *compare_flags,
                ]
            )
        else:
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
                    "PYTHONPATH=/app",
                    "--local_authorities",
                    *chunk,
                    "--release_date",
                    release_date,
                    *compare_flags,
                ]
            )

    # Aggregate any error logs and save to a single file
    dir_uri = Path(
        config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date, local_authorities="REMOVE"
        )
    ).parent
    failure_paths = s3_utils.list_files_s3_path(dir_uri=str(dir_uri))
    if failure_paths:
        failures = pl.concat([pl.read_csv(f) for f in failure_paths])
        save_as = config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date, local_authorities="ORBIT_CONCAT"
        )
        failures.write_csv(save_as)
        print(f"Pipeline failed for some local authorities. See logs at {save_as}")
    else:
        print("Pipeline finished for all Local Authorities successfully!")
