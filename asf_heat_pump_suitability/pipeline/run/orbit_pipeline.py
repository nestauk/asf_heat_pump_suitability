import math
import numpy as np
import argparse
from typing import List


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
        required=False,
        type=int,
    )

    parser.add_argument(
        "--number_of_chunks",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import polars as pl
    from tqdm import tqdm
    import subprocess
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()
    local_authorities = args.local_authorities
    release_date = save_utils.get_str_release_date(args.release_date)
    print(f"Release date pinned to: {release_date}")

    if local_authorities == "GB":
        local_authorities = pl.read_csv(config["data"]["processed"]["valid_la_names"])[
            "LAD23NM"
        ].to_list()

    chunks = chunk_list_strings(
        np.asarray(local_authorities), n=args.size, size=args.number_of_chunks
    )

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
                "PYTHONPATH=/app",
                "--local_authorities",
                *chunk,
                "--release_date",
                release_date,
            ]
        )
