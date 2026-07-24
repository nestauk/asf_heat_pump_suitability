"""
Build and save run manifests recording pipeline output lineage.

Each pipeline entrypoint writes a companion `{output_basename}.manifest.json`
next to every output it saves to S3, recording which input versions, git
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
    Get the git commit hash of the checked-out code.

    Runs in the project directory so the hash reflects the imported package,
    not the caller's working directory.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
    """
    try:
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
    Resolve config["data"] key paths to their dataset path strings, whose
    dated prefixes carry the input versions.

    Args:
        input_keys (list[str]): dot-separated key paths into `config["data"]`,
            typically one of the `STAGE_INPUT_KEYS` lists

    Returns:
        dict: mapping of each key to its path string

    Raises:
        KeyError: if a key is missing or resolves to a config subtree — a typo
            in a curated list must fail loudly, not silently omit lineage
    """
    input_versions = {}
    for dot_key in input_keys:
        value = config["data"]
        for part in dot_key.split("."):
            try:
                value = value[part]
            except (KeyError, TypeError) as error:
                raise KeyError(
                    f"Run manifest input key '{dot_key}' not found in config['data']"
                ) from error
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
    Generate run manifest dict recording the lineage of one pipeline output.

    Args:
        stage (str): pipeline entrypoint that produced the output, e.g. "uprns"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): CLI arguments the entrypoint was run with
        input_keys (list[str], optional): config["data"] key paths of the
            datasets the stage reads; defaults to the stage's curated list in
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

    Convenience wrapper for the pipeline entrypoints, combining
    `generate_dict_run_manifest` and `save_manifest_to_s3`.

    Args:
        output_path (str): S3 path of the output file the manifest describes
        stage (str): pipeline entrypoint that produced the output, e.g. "uprns"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): CLI arguments the entrypoint was run with
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

    Write failures are logged and swallowed: lineage degrades rather than
    aborting a pipeline run.

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
