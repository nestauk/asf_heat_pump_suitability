"""
Runs the entire pipeline for a list of local authorities. Executes each of the five stages in sequence, per local
authority, and checks for errors after each stage.

Usage:
    python asf_heat_pump_suitability/pipeline/run/pipeline.py --local_authorities <LOCAL_AUTHORITY> [<LOCAL_AUTHORITY> ...] [--release_date YYYYMMDD] [--compare [--trigger TRIGGER]]

The release date defaults to today and is pinned across all stages, so a run crossing midnight still writes to a single
dated release directory.

With `--compare`, each local authority whose stages all succeed then gets a cross-version comparison of each stage's
output against its latest earlier version, saved to S3 (see `pipeline/validate/compare_versions.py`). `--trigger`
(`methodology_change` or `input_release`) is passed to every comparison. A failed comparison is listed at the end of
the run but does not fail the local authority.
"""

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

from asf_heat_pump_suitability import config
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

COMPARE_SCRIPT = "asf_heat_pump_suitability/pipeline/validate/compare_versions.py"
# The stages compare_versions can compare, in run order
COMPARE_STAGES = list(config["compare_versions"]["stage_output_datasets"])


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

    parser.add_argument(
        "--compare",
        help="Compare each stage's output with its latest earlier version once a local authority's stages succeed.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--trigger",
        help="Why the run is made; passed to every comparison to check its tolerances. Omit to show the numbers only.",
        choices=sorted(config["compare_versions"]["tolerances"]),
        default=None,
    )

    args = parser.parse_args()
    if args.trigger is not None and not args.compare:
        parser.error("--trigger is used only by the comparisons; add --compare.")
    return args


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


def run_comparison(
    stage: str, local_authority: str, release_date: str, trigger: str | None = None
) -> int:
    """
    Compare one stage's output for one local authority with its latest earlier version, and save the results to S3.

    Args:
        stage (str): stage name as used by `compare_versions.py`, e.g. "decision_tree".
        local_authority (str): local authority name to compare outputs for.
        release_date (str): release date of the new outputs.
        trigger (str | None): why the run is made, passed on as `--trigger`. Default None for no tolerance checks.

    Returns:
        int: the comparison's process exit code (0 on success, including when there is no earlier version).
    """
    trigger_flags = ["--trigger", trigger] if trigger else []
    result = subprocess.run(
        [
            sys.executable,
            COMPARE_SCRIPT,
            "--stage",
            stage,
            "--local_authority",
            local_authority,
            "--new_release_date",
            release_date,
            "--save",
            *trigger_flags,
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

    # (local authority, stage) pairs; kept apart from `failed` so the stage-failure CSV only lists failed stages
    comparisons_failed = []

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

        if args.compare:
            for stage in COMPARE_STAGES:
                print(f"\n--> Comparing: {stage}")
                returncode = run_comparison(
                    stage=stage,
                    local_authority=la,
                    release_date=release_date,
                    trigger=args.trigger,
                )
                if returncode != 0:
                    print(
                        f"Error comparing {stage} for {la}. Continuing; the comparison does not fail the local authority."
                    )
                    comparisons_failed.append((la, stage))

        elapsed = int(time.monotonic() - la_start)
        print(f"Successfully finished pipeline for: {la} (took {elapsed}s)")
        succeeded += 1

    print(
        f"\n\nPipeline completed for {succeeded} of {len(args.local_authorities)} "
        "local authorities."
    )
    if len(failed) > 0:
        print(
            f"Pipeline failed for the following Local Authorities at the specified stages:\n{failed}"
        )
        failed.to_csv(
            config["output"]["log"]["pipeline_failure"].format(
                release_date=release_date
            ),
            mode="a",
            index=False,
            header=False,
        )
    if comparisons_failed:
        print("Comparisons failed for the following Local Authorities and stages:")
        for la, stage in comparisons_failed:
            print(f"  {la}: {stage}")
