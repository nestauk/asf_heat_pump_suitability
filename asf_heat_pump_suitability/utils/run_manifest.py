"""
Functions to build and save run manifests recording pipeline output lineage.

Each pipeline entrypoint writes a companion `{output_basename}.manifest.json`
next to every output it saves to S3, recording which input versions, git
commit and parameters produced that output. The `.manifest.json` suffix keeps
the output basename, so it never collides with the front-end `manifest.json`
written by pipeline/run/create_manifest.py.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone

import fsspec

from asf_heat_pump_suitability import PROJECT_DIR, config

MANIFEST_SUFFIX = ".manifest.json"
UNKNOWN_GIT_COMMIT = "unknown"


def get_str_git_commit() -> str:
    """
    Get the git commit hash of the currently checked-out code.

    Runs `git rev-parse HEAD` in the project directory so the hash reflects
    the imported package, not the caller's working directory.

    Returns:
        str: 40-character commit hash, or "unknown" if git is unavailable
            (e.g. a deployed environment without a .git directory)
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


def generate_dict_input_versions(data_config: dict | None = None) -> dict:
    """
    Generate flat dict of input dataset versions from nested dataset path config.

    Records the raw resolved path strings (whose dated prefixes carry the input
    versions) under dot-separated keys, e.g. `{"epc.domestic": "s3://..."}`.

    Args:
        data_config (dict | None): nested mapping of dataset names to path strings.
            Defaults to `config["data"]`, the raw input paths pinned for this run.

    Returns:
        dict: flat mapping of dot-separated dataset key to path string
    """
    if data_config is None:
        data_config = config["data"]
    input_versions = {}
    for key, value in data_config.items():
        if isinstance(value, dict):
            input_versions.update(
                {
                    f"{key}.{subkey}": path
                    for subkey, path in generate_dict_input_versions(value).items()
                }
            )
        else:
            input_versions[key] = value
    return input_versions


def generate_dict_run_manifest(
    stage: str,
    local_authority: str,
    row_count: int,
    params: dict,
) -> dict:
    """
    Generate run manifest dict recording the lineage of one pipeline output.

    Args:
        stage (str): name of the pipeline entrypoint that produced the output,
            e.g. "uprns" or "decision_tree"
        local_authority (str): local authority slug the output was generated for
        row_count (int): number of rows (or geojson features) in the output
        params (dict): CLI arguments the entrypoint was run with

    Returns:
        dict: run manifest with keys stage, local_authority, run_at, git_commit,
            input_versions, row_count, params
    """
    return {
        "stage": stage,
        "local_authority": local_authority,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_str_git_commit(),
        "input_versions": generate_dict_input_versions(),
        "row_count": row_count,
        "params": params,
    }


def get_str_manifest_path(output_path: str) -> str:
    """
    Get the S3 path of the run manifest for an output file.

    The manifest is co-located with the output and named
    `{output_basename}.manifest.json`, e.g. `plymouth_domestic_uprns.parquet`
    -> `plymouth_domestic_uprns.manifest.json`.

    Args:
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        str: S3 path of the companion run manifest
    """
    return output_path.rsplit(".", 1)[0] + MANIFEST_SUFFIX


def save_manifest_to_s3(manifest: dict, output_path: str) -> None:
    """
    Save run manifest as JSON next to the output file it describes.

    Args:
        manifest (dict): run manifest, as returned by `generate_dict_run_manifest`
        output_path (str): S3 path of the output file the manifest describes

    Returns:
        None
    """
    manifest_path = get_str_manifest_path(output_path)
    logging.info(f"Saving run manifest to {manifest_path}")
    with fsspec.open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
