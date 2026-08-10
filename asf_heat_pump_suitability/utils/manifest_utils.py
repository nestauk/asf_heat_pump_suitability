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
import subprocess
from datetime import datetime, timezone

import fsspec

from asf_heat_pump_suitability import PROJECT_DIR, config

MANIFEST_SUFFIX = ".manifest.json"
UNKNOWN_GIT_COMMIT = "unknown"

STAGE_INPUT_KEYS = config["run_manifest"]["stage_input_keys"]


def get_str_git_commit() -> str:
    """
    Get the git commit hash of the currently checked-out code.

    The git command runs inside this repo's own directory (`cwd=PROJECT_DIR`),
    not wherever the pipeline was launched from. Pipeline scripts can be run
    from anywhere (another folder, a notebook, a scheduled job), and we always
    want the commit of this repo's code, not of whatever directory the caller
    happens to be in.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
    """
    try:
        # capture_output=True collects what the command prints into .stdout
        # (rather than printing it to the terminal) so the hash can be read;
        # check=True raises if git exits with an error.
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        logging.warning(
            "Could not read git commit hash; recording '%s' in run manifest.",
            UNKNOWN_GIT_COMMIT,
        )
        return UNKNOWN_GIT_COMMIT


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
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_str_git_commit(),
        "input_versions": generate_dict_input_versions(input_keys),
        "row_count": row_count,
        "params": params,
    }


def save_run_manifest_to_s3(
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
    return output_path.rsplit(".", 1)[0] + MANIFEST_SUFFIX


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
