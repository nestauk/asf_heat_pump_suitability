"""
Runs the entire pipeline for a list of local authorities. Executes each of the five stages in sequence, per local
authority, and checks for errors after each stage.

Usage:
    python asf_heat_pump_suitability/pipeline/run/run_pipeline.py --local_authorities <LOCAL_AUTHORITY> [<LOCAL_AUTHORITY> ...] [--release_date YYYYMMDD]

The release date defaults to today and is pinned across all stages, so a run crossing midnight still writes to a single
dated release directory.
"""

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

from asf_heat_pump_suitability.utils import save_utils

# (display name, path relative to the repo root) for each stage, in run order
STAGES = [
    ("uprns.py", "asf_heat_pump_suitability/pipeline/transform/uprns.py"),
    ("add_features.py", "asf_heat_pump_suitability/pipeline/run/add_features.py"),
    (
        "decision_tree.py",
        "asf_heat_pump_suitability/pipeline/transform/decision_tree.py",
    ),
    ("cluster.py", "asf_heat_pump_suitability/pipeline/cluster/cluster.py"),
    (
        "compute_contextual_features.py",
        "asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py",
    ),
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_authorities",
        help="One or more local authority names to run the pipeline for, "
        "e.g. --local_authorities 'Plymouth' 'Vale of Glamorgan'.",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--release_date",
        help="The release date to pin across all stages, e.g. 20260801. "
        "Defaults to today.",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--prod",
        help="Set to push changes to production (i.e. staging area). This should only be used when running from `dev`.",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def run_script(
    script_path: str, local_authority: str, release_date: str, flags: list = None
) -> int:
    """
    Run one pipeline stage for one local authority and save outputs to S3.

    Args:
        script_path (str): path to the stage script, relative to the repo root.
        local_authority (str): local authority name to run the stage for.
        release_date (str): release date to pin, in the configured format.
        flags (list): additional flags for running pipeline script. Default None.

    Returns:
        int: the stage's process exit code (0 on success).
    """
    flags = flags or []
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--local_authorities",
            local_authority,
            "--release_date",
            release_date,
            "--save",
            *flags,
        ]
    )
    return result.returncode


if __name__ == "__main__":
    # Run from the repo root regardless of where the script is invoked from
    from asf_heat_pump_suitability import PROJECT_DIR, config

    os.chdir(PROJECT_DIR)

    args = parse_arguments()
    release_date = save_utils.get_str_release_date(args.release_date)
    print(f"Release date pinned to: {release_date}")

    # TODO commented out due to errors in running check_inputs.py
    # print("--> Checking S3 input paths exist: check_inputs.py")
    # result = subprocess.run(
    #     [
    #         sys.executable,
    #         "asf_heat_pump_suitability/pipeline/validate/check_inputs.py",
    #         "--local_authorities",
    #         *args.local_authorities,
    #     ]
    # )
    # if result.returncode != 0:
    #     sys.exit("Error running check_inputs.py: missing S3 input paths. Aborting.")

    failed = pd.DataFrame({"local_authority": [], "stage_failed": []})

    succeeded = 0
    for la in args.local_authorities:
        print("=" * 50)
        print(f"Starting pipeline for: {la}")
        print("=" * 50)
        la_start = time.monotonic()

        la_failed = False
        for stage_name, script_path in STAGES:
            print(f"\n--> Running: {stage_name}")
            if stage_name == "compute_contextual_features.py" and args.prod:
                returncode = run_script(script_path, la, release_date, flags=["--prod"])
            else:
                returncode = run_script(script_path, la, release_date)
            if returncode != 0:
                elapsed = int(time.monotonic() - la_start)
                print(
                    f"Error in {stage_name} for {la}. Skipping... "
                    f"(after {elapsed}s)"
                )
                la_failed = True
                failed["local_authority"] = la
                failed["stage_failed"] = stage_name
                break

        if la_failed:
            continue

        elapsed = int(time.monotonic() - la_start)
        print(f"Successfully finished pipeline for: {la} (took {elapsed}s)")
        succeeded += 1

    print(
        f"\n\nPipeline completed for {succeeded} of {len(args.local_authorities)} "
        "local authorities."
    )
    if len(failed) > 0:
        failure_fpath = config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date, local_authorities=args.local_authorities
        )
        print(
            f"Pipeline failed for the following Local Authorities at the specified stages:\n{failed}."
            f"\nSee which stage each failure occurred in: {failure_fpath}"
        )
        failed.to_csv(
            failure_fpath,
            mode="a",
            index=False,
            header=False,
        )
