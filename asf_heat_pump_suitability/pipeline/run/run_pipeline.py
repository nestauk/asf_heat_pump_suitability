"""
Runs the entire pipeline for a list of local authorities. Executes each of the five stages in sequence, per local
authority, and checks for errors after each stage.

You can then run it with:
    asf_heat_pump_suitability/pipeline/run/run_pipeline.py --local_authorities "Plymouth" "Vale of Glamorgan"

Usage:
    python asf_heat_pump_suitability/pipeline/run/run_pipeline.py --local_authorities <LOCAL_AUTHORITY> [--release_date YYYYMMDD]

The release date defaults to today and is pinned across all stages, so a run crossing midnight still writes to a single
dated release directory.
"""

import argparse
import os
import subprocess
import sys
import time

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
    return parser.parse_args()


def run_script(script_path: str, local_authority: str, release_date: str) -> int:
    """
    Run one pipeline stage for one local authority and save outputs to S3.

    Args:
        script_path: path to the stage script, relative to the repo root.
        local_authority: local authority name to run the stage for.
        release_date: release date to pin, in the configured format.

    Returns:
        int: the stage's process exit code (0 on success).
    """
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--local_authorities",
            local_authority,
            "--release_date",
            release_date,
            "--save",
        ]
    )
    return result.returncode


if __name__ == "__main__":
    # Run from the repo root regardless of where the script is invoked from
    from asf_heat_pump_suitability import PROJECT_DIR

    os.chdir(PROJECT_DIR)

    args = parse_arguments()
    release_date = save_utils.get_str_release_date(args.release_date)
    print(f"Release date pinned to: {release_date}")

    print("--> Checking S3 input paths exist: check_inputs.py")
    result = subprocess.run(
        [
            sys.executable,
            "asf_heat_pump_suitability/pipeline/validate/check_inputs.py",
            "--local_authorities",
            *args.local_authorities,
        ]
    )
    if result.returncode != 0:
        sys.exit("Error running check_inputs.py: missing S3 input paths. Aborting.")

    succeeded = 0
    for la in args.local_authorities:
        print("=" * 50)
        print(f"Starting pipeline for: {la}")
        print("=" * 50)
        la_start = time.monotonic()

        la_failed = False
        for stage_name, script_path in STAGES:
            print(f"\n--> Running: {stage_name}")
            returncode = run_script(script_path, la, release_date)
            if returncode != 0:
                elapsed = int(time.monotonic() - la_start)
                print(
                    f"Error in {stage_name} for {la}. Skipping... "
                    f"(after {elapsed}s)"
                )
                la_failed = True
                break

        if la_failed:
            continue

        elapsed = int(time.monotonic() - la_start)
        print(f"Successfully finished pipeline for: {la} (took {elapsed}s)")
        succeeded += 1
        print()

    print(
        f"Pipeline completed for {succeeded} of {len(args.local_authorities)} "
        "local authorities."
    )
    print("=" * 50)

    # print("--> Generating manifest.json: create_manifest.py")
    # result = subprocess.run(
    #     [sys.executable, "asf_heat_pump_suitability/pipeline/run/create_manifest.py"]
    # )
    # if result.returncode != 0:
    #     sys.exit("Error running create_manifest.py")

    print("Pipeline finished!")
