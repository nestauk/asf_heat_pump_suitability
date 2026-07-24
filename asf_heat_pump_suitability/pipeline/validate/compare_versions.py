"""
Compare two dated versions of a pipeline stage output for one local authority.

Reports row/UPRN count deltas, schema diff, UPRN churn, the tech-assignment
transition matrix (decision-tree stage), and a module-scoped commit log
between the two versions' recorded commits, read against the tolerance rubric
of the trigger that prompted the comparison. Writes a local markdown report
and logs a console summary.

Usage:
python -m asf_heat_pump_suitability.pipeline.validate.compare_versions \
    --stage decision_tree \
    --local_authority plymouth \
    --old_release_date 20260601 \
    --new_release_date 20260722 \
    --trigger methodology_change
"""

import json
import logging
import subprocess

import fsspec
import polars as pl

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.utils import manifest_utils

UPRN_COL = "UPRN"
TECH_COL = "assigned_tech"

STAGE_MODULE_PATHS = config["compare_versions"]["stage_module_paths"]
TOLERANCES = config["compare_versions"]["tolerances"]


def get_dict_tolerances(trigger: str) -> dict:
    """
    Get the tolerances for a comparison trigger's rubric from base.yaml.

    Args:
        trigger: why the comparison is being run, e.g. "methodology_change"

    Returns:
        dict: tolerance name to value

    Raises:
        KeyError: if no rubric is configured for the trigger
    """
    try:
        return TOLERANCES[trigger]
    except KeyError as error:
        raise KeyError(
            f"No tolerance rubric configured for trigger '{trigger}'; expected "
            f"one of {sorted(TOLERANCES)} in config['compare_versions']"
        ) from error


def generate_dict_count_delta(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict:
    """
    Compare row and distinct-UPRN counts between two versions of an output.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: old/new/delta counts; UPRN entries are None when either version
            has no UPRN column (e.g. cluster-level outputs)
    """
    counts = {
        "rows_old": df_old.height,
        "rows_new": df_new.height,
        "rows_delta": df_new.height - df_old.height,
        "uprns_old": None,
        "uprns_new": None,
        "uprns_delta": None,
    }
    if UPRN_COL in df_old.columns and UPRN_COL in df_new.columns:
        counts["uprns_old"] = df_old[UPRN_COL].n_unique()
        counts["uprns_new"] = df_new[UPRN_COL].n_unique()
        counts["uprns_delta"] = counts["uprns_new"] - counts["uprns_old"]
    return counts


def generate_dict_schema_diff(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict:
    """
    Diff column names and dtypes between two versions of an output.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: "added" and "removed" map column to dtype string;
            "dtype_changed" maps column to an (old, new) dtype string pair
    """
    schema_old = {col: str(dtype) for col, dtype in df_old.schema.items()}
    schema_new = {col: str(dtype) for col, dtype in df_new.schema.items()}
    return {
        "added": {col: schema_new[col] for col in schema_new if col not in schema_old},
        "removed": {
            col: schema_old[col] for col in schema_old if col not in schema_new
        },
        "dtype_changed": {
            col: (schema_old[col], schema_new[col])
            for col in schema_old
            if col in schema_new and schema_old[col] != schema_new[col]
        },
    }


def generate_dict_uprn_churn(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict | None:
    """
    Count UPRNs added, removed and retained between two versions of an output.

    UPRNs are compared as strings so a dtype change between versions does not
    read as full churn.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: added/removed/retained counts and the removed share of old
            UPRNs, or None when either version has no UPRN column
    """
    if UPRN_COL not in df_old.columns or UPRN_COL not in df_new.columns:
        return None
    uprns_old = set(df_old[UPRN_COL].cast(pl.Utf8).to_list())
    uprns_new = set(df_new[UPRN_COL].cast(pl.Utf8).to_list())
    n_removed = len(uprns_old - uprns_new)
    return {
        "n_added": len(uprns_new - uprns_old),
        "n_removed": n_removed,
        "n_retained": len(uprns_old & uprns_new),
        "removed_share": n_removed / len(uprns_old) if uprns_old else 0.0,
    }


def generate_str_churn_note(churn: dict, max_removed_share: float) -> str | None:
    """
    Flag UPRN loss above the rubric's tolerance.

    Args:
        churn: churn counts, as returned by `generate_dict_uprn_churn`
        max_removed_share: rubric tolerance for the removed share of old UPRNs

    Returns:
        str: warning naming the observed share and the tolerance, or None
            when the churn is within tolerance
    """
    if churn["removed_share"] <= max_removed_share:
        return None
    return (
        f"WARNING: {churn['removed_share']:.1%} of old UPRNs were removed, above "
        f"the {max_removed_share:.1%} tolerance for this rubric."
    )


def generate_df_tech_transitions(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> pl.DataFrame:
    """
    Count tech-assignment transitions for UPRNs present in both versions.

    Args:
        df_old: older version of the UPRN-level decision-tree output
        df_new: newer version of the UPRN-level decision-tree output

    Returns:
        pl.DataFrame: one row per (assigned_tech_old, assigned_tech_new) pair
            with its UPRN count, largest first
    """
    old = df_old.select(
        pl.col(UPRN_COL).cast(pl.Utf8),
        pl.col(TECH_COL).alias("assigned_tech_old"),
    )
    new = df_new.select(
        pl.col(UPRN_COL).cast(pl.Utf8),
        pl.col(TECH_COL).alias("assigned_tech_new"),
    )
    return (
        old.join(new, on=UPRN_COL, how="inner")
        .group_by("assigned_tech_old", "assigned_tech_new")
        .agg(pl.len().alias("n_uprns"))
        .sort("n_uprns", "assigned_tech_old", "assigned_tech_new", descending=True)
    )


def load_dict_manifest(output_path: str) -> dict | None:
    """
    Load the run manifest saved next to a pipeline output.

    Args:
        output_path: S3 path of the output file the manifest describes

    Returns:
        dict: the run manifest, or None when it is missing or unreadable
            (outputs predating the run manifest have none)
    """
    manifest_path = manifest_utils.get_str_manifest_path(output_path)
    try:
        with fsspec.open(manifest_path) as f:
            return json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        logging.warning("No readable run manifest at %s", manifest_path)
        return None


def generate_dict_input_version_changes(manifest_old: dict, manifest_new: dict) -> dict:
    """
    Diff the input dataset versions recorded in two run manifests.

    Args:
        manifest_old: run manifest of the older output version
        manifest_new: run manifest of the newer output version

    Returns:
        dict: "changed" maps input key to an (old, new) path pair; "added"
            and "removed" map inputs recorded by only one manifest to their path
    """
    versions_old = manifest_old["input_versions"]
    versions_new = manifest_new["input_versions"]
    return {
        "changed": {
            key: (versions_old[key], versions_new[key])
            for key in versions_old
            if key in versions_new and versions_old[key] != versions_new[key]
        },
        "added": {
            key: path for key, path in versions_new.items() if key not in versions_old
        },
        "removed": {
            key: path for key, path in versions_old.items() if key not in versions_new
        },
    }


def generate_list_commit_log(
    commit_old: str, commit_new: str, stage: str
) -> list[str] | None:
    """
    List commits between two recorded commits that touched a stage's modules.

    Args:
        commit_old: git commit recorded in the older version's manifest
        commit_new: git commit recorded in the newer version's manifest
        stage: pipeline stage whose curated `STAGE_MODULE_PATHS` scope the log

    Returns:
        list[str]: one "short-hash subject" line per commit, or None when a
            recorded commit is the "unknown" sentinel or absent from local
            git history (e.g. an unfetched branch)
    """
    if manifest_utils.UNKNOWN_GIT_COMMIT in (commit_old, commit_new):
        logging.warning("A recorded commit is unknown; cannot build a commit log.")
        return None
    if commit_old == commit_new:
        return []
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--oneline",
                f"{commit_old}..{commit_new}",
                "--",
                *STAGE_MODULE_PATHS[stage],
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        logging.warning(
            "git log %s..%s failed; are both commits fetched locally?",
            commit_old,
            commit_new,
        )
        return None
    return result.stdout.splitlines()
