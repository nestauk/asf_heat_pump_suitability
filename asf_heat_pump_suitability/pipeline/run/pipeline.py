"""
Runs the entire pipeline for a list of local authorities. Executes each of the five stages in sequence, per local
authority, and checks for errors after each stage. Saves outputs to S3.

Usage:
    python asf_heat_pump_suitability/pipeline/run/pipeline.py --local_authorities <LOCAL_AUTHORITY> [<LOCAL_AUTHORITY> ...] --model_release YYYYMMDD [--release_date YYYYMMDD] [--compare [--trigger TRIGGER] [--old_release_date YYYYMMDD]]

Set --model_release to the YYYYMMDD date of the block of flats model training run. This is used to load the trained
block of flats classifier and the manually labelled data it was trained on from the corresponding dated directory.

The release date defaults to today and is pinned across all stages, so a run crossing midnight still writes to a single
dated release directory.

With `--compare`, each local authority whose stages all succeed then gets a cross-version comparison of each stage's
output against its latest earlier version, saved to S3 (see `pipeline/validate/compare_versions.py`). `--trigger`
says why the run is made, and is passed to every comparison to pick its tolerances:
- `methodology_change`: the code or the method changed, and the inputs did not.
- `input_release`: a new release of an input dataset, and the code did not change.
`--old_release_date` makes every comparison
use that release instead, for example if you wanted to test against an earlier version than the most recent run. A stage with no
output at that date fails its comparison; there is no fallback to another date. A failed comparison is listed at the
end of the run but does not fail the local authority.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from asf_heat_pump_suitability import PROJECT_DIR, config
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
# The stages that compare_versions.py should compare, in run order
COMPARE_STAGES = list(config["compare_versions"]["stage_output_datasets"])


def add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the comparison flags. `orbit_pipeline.py` uses them too and passes them on to every chunk.

    Args:
        parser (argparse.ArgumentParser): the launcher's argument parser
    """
    parser.add_argument(
        "--compare",
        help="Compare each stage's output with its latest earlier version once a local authority's stages succeed.",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--trigger",
        help="Why the run is made; passed to every comparison to check its tolerances. Omit to show the numbers only.",
        choices=sorted(config["compare_versions"]["tolerances"]),
        default=None,
        required=False,
    )
    parser.add_argument(
        "--old_release_date",
        help="Compare every stage against this release, e.g. 20260901, instead of each stage's latest earlier one.",
        type=save_utils.get_str_release_date,
        default=None,
        required=False,
    )


def check_compare_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """
    Stop with a usage error when a comparison flag is used wrongly.

    `--trigger` and `--old_release_date` need `--compare`, and
    `--old_release_date` must be earlier than the release date.

    Args:
        parser (argparse.ArgumentParser): the launcher's argument parser
        args (argparse.Namespace): the parsed arguments, with `release_date`
            and the flags from `add_compare_arguments`
    """
    if args.trigger is not None and not args.compare:
        parser.error(
            "--trigger is used only by the comparisons; add --compare to run with comparison."
        )
    if args.old_release_date is not None:
        if not args.compare:
            parser.error(
                "--old_release_date is used only by the comparisons; add --compare to run with comparison."
            )
        try:
            release_date = save_utils.get_str_release_date(args.release_date)
        except ValueError as error:
            parser.error(str(error))
        if args.old_release_date >= release_date:
            parser.error("--old_release_date must be earlier than the release date.")


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
        "--model_release",
        help="Date of model training in YYYYMMDD format.",
        type=str,
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

    add_compare_arguments(parser=parser)

    args = parser.parse_args()
    check_compare_arguments(parser=parser, args=args)
    return args


def scrub_str_local_paths(text: str) -> str:
    """
    Replace local absolute paths in `text` with placeholders, so error logs don't expose usernames.
    The project directory becomes `<PROJECT_DIR>` and the user's home directory becomes `~`.

    Args:
        text (str): text which may contain local file paths, e.g. an error traceback.

    Returns:
        str: `text` with local paths replaced.
    """
    return text.replace(str(PROJECT_DIR), "<PROJECT_DIR>").replace(
        str(Path.home()), "~"
    )


def run_script(
    script_path: str, local_authority: str, release_date: str, flags: list = None
) -> tuple[int, str, list[str]]:
    """
    Run one pipeline stage for one local authority and save outputs to S3.

    Args:
        script_path (str): path to the stage script, relative to the repo root.
        local_authority (str): local authority name to run the stage for.
        release_date (str): release date to pin, in the configured format.
        flags (list): additional flags for running pipeline script. Default None.

    Returns:
        tuple[int, str, list[str]]: the stage's process exit code (0 on success), the error traceback if it failed
        (otherwise None), and a list of every warning the stage printed to stderr with the `VALIDATION_WARNING:` prefix
        (empty if None).
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
        ],
        stderr=subprocess.PIPE,
        text=True,
    )

    traceback = (
        scrub_str_local_paths(result.stderr.strip()) if result.returncode != 0 else None
    )

    validation_warnings = [
        scrub_str_local_paths(line.removeprefix("VALIDATION_WARNING:").strip())
        for line in result.stderr.splitlines()
        if line.startswith("VALIDATION_WARNING:")
    ]

    return result.returncode, traceback, validation_warnings


def run_comparison(
    stage: str,
    local_authority: str,
    release_date: str,
    trigger: str | None = None,
    old_release_date: str | None = None,
) -> int:
    """
    Compare one stage's output for one local authority with an earlier version, and save the results to S3.

    Args:
        stage (str): stage name as used by `compare_versions.py`, e.g. "decision_tree".
        local_authority (str): local authority name to compare outputs for.
        release_date (str): release date of the new outputs.
        trigger (str | None): why the run is made, passed on as `--trigger`: "methodology_change" or
            "input_release". Default None for no tolerance checks.
        old_release_date (str | None): release to compare against. Default None for the stage's latest earlier one.

    Returns:
        int: the comparison's process exit code (0 on success, including when there is no earlier version).
    """
    trigger_flags = ["--trigger", trigger] if trigger else []
    old_flags = ["--old_release_date", old_release_date] if old_release_date else []
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
            *old_flags,
        ]
    )
    return result.returncode


if __name__ == "__main__":
    from asf_heat_pump_suitability.pipeline.transform import local_authority

    # Run from the repo root regardless of where the script is invoked from
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

    failed = {
        "local_authority": [],
        "stage_failed": [],
        "level": [],  # e.g. "error" or "warning"
        "error_message": [],
    }

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
                returncode, error_message, validation_warnings = run_script(
                    script_path=script_path,
                    local_authority=la,
                    release_date=release_date,
                    flags=["--prod"],
                )
            elif stage_name == "add_features.py":
                returncode, error_message, validation_warnings = run_script(
                    script_path=script_path,
                    local_authority=la,
                    release_date=release_date,
                    flags=["--model_release", args.model_release],
                )
            else:
                returncode, error_message, validation_warnings = run_script(
                    script_path=script_path,
                    local_authority=la,
                    release_date=release_date,
                )

            for warning in validation_warnings:
                failed["local_authority"].append(la)
                failed["stage_failed"].append(stage_name)
                failed["level"].append("warning")
                failed["error_message"].append(warning)

            if returncode != 0:
                elapsed = int(time.monotonic() - la_start)
                print(
                    f"Error in {stage_name} for {la}. Skipping... "
                    f"(after {elapsed}s)"
                )
                la_failed = True
                failed["local_authority"].append(la)
                failed["stage_failed"].append(stage_name)
                failed["level"].append("error")
                failed["error_message"].append(error_message)
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
                    old_release_date=args.old_release_date,
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

    if len(failed["local_authority"]) > 0:
        local_authority_dict = local_authority.get_dict_la_data(
            failed["local_authority"]
        )
        failed = pd.DataFrame(failed)
        failure_fpath = config["output"]["log"]["pipeline_failure"].format(
            release_date=release_date,
            local_authorities=local_authority_dict["url_slug"],
        )
        if (failed["level"] == "error").any():
            print(
                f"Pipeline failed for some local authorities. See logs at {failure_fpath}"
            )
        else:
            print(
                f"Pipeline finished for all local authorities, some with warnings. See logs at {failure_fpath}"
            )
        failed.to_csv(failure_fpath, index=False)
    if comparisons_failed:
        print("Comparisons failed for the following Local Authorities and stages:")
        for la, stage in comparisons_failed:
            print(f"  {la}: {stage}")
