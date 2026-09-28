import math
import numpy as np
import argparse


def chunk_list_local_authorities(arr: np.array, n: int, size: bool) -> list:
    if not size:
        # Calculate how many items per chunk
        n = math.ceil(len(arr) / n)
    chunks = [f'"{'" "'.join(l)}"' for l in np.array_split(arr, n)]
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
        local_authorities = pl.read_csv(config["data"]["processed"]["valid_la_names"])

    chunks = chunk_list_local_authorities(
        np.asarray(local_authorities), n=args.size, size=args.number_of_chunks
    )

    for chunk in tqdm(chunks):
        print(f"Running chunk: {chunk}")
        subprocess.run(
            [
                "orbit",
                "launch",
                "--script",
                "asf_heat_pump_suitability/pipeline/run/run_pipeline.py",
                "--team",
                "ASF",
                "--project",
                "local_heat_planning_tool",
                "--cpu",
                "8",
                "--memory",
                "60gb",
                "-e",
                "PYTHONPATH=/app",
                "--local_authorities",
                chunk,
                "--release_date",
                release_date,
            ]
        )
