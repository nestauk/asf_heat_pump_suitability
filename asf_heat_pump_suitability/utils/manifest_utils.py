"""
Build and save run manifests recording how each pipeline output was produced.

Each pipeline script writes a companion `{output_basename}.manifest.json`
next to every output it saves to S3, recording which input datasets, git
commit and parameters produced it. Keeping the output basename in the filename
avoids colliding with the front-end `manifest.json` from
pipeline/run/create_manifest.py.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone

import fsspec

from asf_heat_pump_suitability import PROJECT_DIR, config

MANIFEST_SUFFIX = ".manifest.json"
UNKNOWN_GIT_COMMIT = "unknown"

STAGE_INPUT_KEYS = config["run_manifest"]["stage_input_keys"]


def run_git_or_none(
    args: list[str], warning: str, *warning_args: object
) -> subprocess.CompletedProcess | None:
    """
    Run a git command in this repo's own directory, or None on failure.

    `cwd=PROJECT_DIR` so callers get this repo's git state no matter where
    the pipeline was launched from (another folder, a notebook, a scheduled
    job). On any git failure the warning is logged and None returned —
    lineage and reporting degrade rather than abort a run.

    Args:
        args: full git command, e.g. ["git", "rev-parse", "HEAD"]
        warning: logging.warning format string logged on failure
        *warning_args: values for the warning format string

    Returns:
        subprocess.CompletedProcess: completed command with captured stdout,
            or None when git exits non-zero or cannot run
    """
    try:
        return subprocess.run(
            args, cwd=PROJECT_DIR, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError):
        logging.warning(warning, *warning_args)
        return None


def get_str_git_commit() -> str:
    """
    Get the git commit hash of the currently checked-out code.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
    """
    result = run_git_or_none(
        ["git", "rev-parse", "HEAD"],
        "Could not read git commit hash; recording '%s' in run manifest.",
        UNKNOWN_GIT_COMMIT,
    )
    return UNKNOWN_GIT_COMMIT if result is None else result.stdout.strip()


def generate_dict_input_versions(input_keys: list[str]) -> dict:
    """
    Look up the S3 path of each input dataset, as a record of which version
    of the data was used.

    Input dataset files in this repo carry a date in their filename or folder
    (e.g. `202412_v1_...`), so the full path doubles as a version record: if
    the path is different between two runs, the input data changed.

    Args:
        input_keys (list[str]): dot-separated key paths in `config["data"]`,
            e.g. "epc.domestic" for `config["data"]["epc"]["domestic"]`;
            typically one of the `STAGE_INPUT_KEYS` lists

    Returns:
        dict: mapping of each key to its dataset path

    Raises:
        KeyError: if a key is missing from `config["data"]`, or stops at a
            group of datasets rather than a single dataset path. A typo in
            the config lists should stop the run with a clear error, not
            quietly leave an input out of the manifest.
    """
    input_versions = {}
    for dot_key in input_keys:
        # Walk down config["data"] one key at a time, e.g. "epc.domestic"
        # -> config["data"]["epc"] -> config["data"]["epc"]["domestic"]
        value = config["data"]
        for part in dot_key.split("."):
            try:
                value = value[part]
            except (KeyError, TypeError) as error:
                raise KeyError(
                    f"Run manifest input key '{dot_key}' not found in config['data']"
                ) from error
        # Reaching a dict means the key stopped at a group of datasets
        # (e.g. "epc") instead of a single dataset path (e.g. "epc.domestic")
        if isinstance(value, dict):
            raise KeyError(
                f"Run manifest input key '{dot_key}' resolves to a config['data'] "
                "subtree, not a dataset path"
            )
        input_versions[dot_key] = value
    return input_versions


def generate_dict_run_manifest(
    stage: str,
    local_authority: str,
    row_count: int,
    params: dict,
    input_keys: list[str] | None = None,
) -> dict:
    """
    Generate run manifest dict recording how one pipeline output was produced.

    Args:
        stage (str): pipeline script that produced the output, e.g. "uprns".
            Each script passes its own name, so every output is labelled with
            the script that made it; the valid names are the keys of
            `STAGE_INPUT_KEYS`.
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): additional CLI arguments the script was run with, as
            {argument name: value}, e.g. {"release_date": "20260722"}
        input_keys (list[str], optional): config["data"] key paths of the
            datasets the script reads; defaults to the script's list in
            `STAGE_INPUT_KEYS`

    Returns:
        dict: run manifest
    """
    if input_keys is None:
        input_keys = STAGE_INPUT_KEYS[stage]
    return {
        "stage": stage,
        "local_authority": local_authority,
        # UTC, matching S3's own file timestamps; the string carries its
        # timezone with it (ends "+00:00")
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_str_git_commit(),
        "input_versions": generate_dict_input_versions(input_keys),
        "row_count": row_count,
        "params": params,
    }


def generate_and_save_run_manifest_to_s3(
    output_path: str,
    stage: str,
    local_authority: str,
    row_count: int,
    params: dict,
) -> None:
    """
    Generate a run manifest and save it next to the output file it describes.

    Convenience wrapper for the pipeline scripts, combining
    `generate_dict_run_manifest` and `save_manifest_to_s3`.

    Args:
        output_path (str): S3 path of the output file the manifest describes
        stage (str): pipeline script that produced the output, e.g. "uprns"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): additional CLI arguments the script was run with, as
            {argument name: value}, e.g. {"release_date": "20260722"}
    """
    manifest = generate_dict_run_manifest(
        stage=stage,
        local_authority=local_authority,
        row_count=row_count,
        params=params,
    )
    save_manifest_to_s3(manifest, output_path)


def get_str_manifest_path(output_path: str) -> str:
    """
    Get the S3 path of the run manifest for an output file.

    Args:
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        str: co-located path ending `.manifest.json`
    """
    # os.path.splitext, not pathlib: Path mangles "s3://" URLs and Path.stem
    # drops the directory. splitext only touches the extension.
    return os.path.splitext(output_path)[0] + MANIFEST_SUFFIX


def save_manifest_to_s3(manifest: dict, output_path: str) -> None:
    """
    Save run manifest as JSON next to the output file it describes.

    If the write fails, the error is logged as a warning and not raised: a
    failed manifest write should never crash a pipeline run whose output has
    already saved successfully. The run just goes without its manifest.

    Args:
        manifest (dict): run manifest, as returned by `generate_dict_run_manifest`
        output_path (str): S3 path of the output file the manifest describes
    """
    manifest_path = get_str_manifest_path(output_path)
    logging.info(f"Saving run manifest to {manifest_path}")
    try:
        with fsspec.open(manifest_path, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.warning(
            "Failed to write run manifest to %s; continuing without it.",
            manifest_path,
            exc_info=True,
        )
