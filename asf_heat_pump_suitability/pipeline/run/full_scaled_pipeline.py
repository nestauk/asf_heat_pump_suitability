import math
import numpy as np
import argparse


def chunk_list(arr: np.array, n: int, size: bool) -> list:
    if not size:
        # Calculate how many items per chunk
        n = math.ceil(len(arr) / n)
    chunks = [list(l) for l in np.array_split(arr, n)]
    print(f"{len(chunks)} chunks to process. Each approx. {len(chunks[0])} items long.")

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
    )

    parser.add_argument(
        "--size",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--number_of_chunks",
        required=False,
        type="store_true",
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    from tqdm import tqdm
    import subprocess

    args = parse_arguments()
    local_authorities = args.local_authorities
    release_date = args.release_date

    chunks = chunk_list(
        np.asarray(local_authorities), n=args.size, size=args.number_of_chunks
    )

    for chunk in tqdm(chunks):
        subprocess.run(
            [
                "orbit",
                "launch",
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
