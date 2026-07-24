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

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import polars as pl
import pyarrow.parquet as pq

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import manifest_utils, save_utils

UPRN_COL = "UPRN"
TECH_COL = "assigned_tech"

STAGE_MODULE_PATHS = config["compare_versions"]["stage_module_paths"]
STAGE_OUTPUT_DATASETS = config["compare_versions"]["stage_output_datasets"]
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


def _expr_uprn_canonical() -> pl.Expr:
    """
    Cast the UPRN column to a canonical string, safe against a numeric
    dtype mismatch between versions (e.g. one side upcast to Float64 by a
    pandas/geopandas round-trip). Going via Float64 then Int64 first means
    an Int64 123 and a Float64 123.0 both land on the string "123", instead
    of "123" vs "123.0" reading as churn.
    """
    return pl.col(UPRN_COL).cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8)


def generate_dict_uprn_churn(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict | None:
    """
    Count UPRNs added, removed and retained between two versions of an output.

    UPRNs are compared as canonical strings so a dtype change between
    versions does not read as full churn.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: added/removed/retained counts and the removed share of old
            UPRNs, or None when either version has no UPRN column
    """
    if UPRN_COL not in df_old.columns or UPRN_COL not in df_new.columns:
        return None
    uprns_old = set(df_old.select(_expr_uprn_canonical()).to_series().to_list())
    uprns_new = set(df_new.select(_expr_uprn_canonical()).to_series().to_list())
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
) -> pl.DataFrame | None:
    """
    Count tech-assignment transitions for UPRNs present in both versions.

    Null tech assignments are labelled "(null)" so they appear as a regular
    matrix row/column. Each version is deduplicated on UPRN first, so a
    duplicate UPRN (a data-quality regression, not expected but not
    prevented upstream either) can't cross-product into inflated counts —
    the count/churn checks already flag a rows-vs-UPRNs mismatch when one
    occurs.

    Args:
        df_old: older version of the UPRN-level decision-tree output
        df_new: newer version of the UPRN-level decision-tree output

    Returns:
        pl.DataFrame: one row per (assigned_tech_old, assigned_tech_new) pair
            with its UPRN count, largest first, or None when either version
            has no tech-assignment column
    """
    if TECH_COL not in df_old.columns or TECH_COL not in df_new.columns:
        return None
    old = df_old.select(
        _expr_uprn_canonical(),
        pl.col(TECH_COL).fill_null("(null)").alias("assigned_tech_old"),
    ).unique(subset=[UPRN_COL], keep="first")
    new = df_new.select(
        _expr_uprn_canonical(),
        pl.col(TECH_COL).fill_null("(null)").alias("assigned_tech_new"),
    ).unique(subset=[UPRN_COL], keep="first")
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
            recorded commit is the "unknown" sentinel, absent from local git
            history (e.g. an unfetched branch), or not an ancestor of the
            newer commit (`old..new` would silently omit its side's commits)
    """
    if manifest_utils.UNKNOWN_GIT_COMMIT in (commit_old, commit_new):
        logging.warning("A recorded commit is unknown; cannot build a commit log.")
        return None
    if commit_old == commit_new:
        return []
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_old, commit_new],
            cwd=PROJECT_DIR,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        logging.warning(
            "%s is not an ancestor of %s (or a commit is unfetched); "
            "old..new would silently omit commits, so the log is skipped.",
            commit_old,
            commit_new,
        )
        return None
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


def get_str_stage_output_path(
    stage: str,
    local_authority: str,
    release_date: str,
    check_exists: bool = False,
) -> str:
    """
    Build the S3 path of the output a stage's comparison reads.

    Args:
        stage: pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority: local authority slug used in output paths
        release_date: dated version folder in YYYYMMDD format
        check_exists: if True, raise when no file exists at the path

    Returns:
        str: S3 path of the stage's output for that version
    """
    return save_utils.get_str_output_path(
        STAGE_OUTPUT_DATASETS[stage],
        release_date=release_date,
        check_exists=check_exists,
        local_authority=local_authority,
        local_authorities=local_authority,
        tolerance_m=config["constant"]["clustering"]["tolerance_m"],
    )


def load_transform_df_stage_output(path: str) -> pl.DataFrame:
    """
    Load a stage output as a plain DataFrame for tabular comparison.

    Geometry columns are dropped: the base checks are tabular, and polars
    cannot read the geoarrow extension columns geopandas-written outputs
    carry. Geojson outputs are loaded with geopandas (EPSG:4326, as saved).
    A geojson with zero features (e.g. every cluster filtered out for a
    local authority) degrades to an empty DataFrame instead of raising.

    Args:
        path: S3 path of the stage output (.parquet or .geojson)

    Returns:
        pl.DataFrame: the output's tabular columns

    Raises:
        ValueError: for file types the comparison cannot read
    """
    if path.endswith(".parquet"):
        schema = pq.read_schema(path)
        tabular = [
            field.name
            for field in schema
            if not (field.metadata or {})
            .get(b"ARROW:extension:name", b"")
            .startswith(b"geoarrow")
        ]
        return pl.from_arrow(pq.read_table(path, columns=tabular))
    if path.endswith(".geojson"):
        try:
            gdf = base_getters.load_gdf_from_s3_geojson(path, crs="EPSG:4326")
        except ValueError:
            logging.warning("No features in geojson at %s; comparing as empty.", path)
            return pl.DataFrame()
        return pl.from_pandas(gdf.drop(columns="geometry"))
    raise ValueError(f"Cannot compare file type of {path}; expected parquet/geojson.")


def _generate_str_counts_section(counts: dict) -> str:
    """Render the row/UPRN count delta as a markdown section."""
    lines = [
        "## Row and UPRN counts",
        "",
        "| Metric | Old | New | Delta |",
        "| --- | --- | --- | --- |",
        (
            f"| Rows | {counts['rows_old']} | {counts['rows_new']} "
            f"| {counts['rows_delta']:+d} |"
        ),
    ]
    if counts["uprns_old"] is None:
        lines.append("\nNo UPRN column in this stage's output.")
    else:
        lines.append(
            f"| Distinct UPRNs | {counts['uprns_old']} | {counts['uprns_new']} "
            f"| {counts['uprns_delta']:+d} |"
        )
    return "\n".join(lines)


def _generate_str_schema_section(schema_diff: dict) -> str:
    """Render the schema diff as a markdown section."""
    lines = ["## Schema diff", ""]
    if not any(schema_diff.values()):
        lines.append("No schema changes.")
    lines.extend(
        f"- Added: `{col}` ({dtype})" for col, dtype in schema_diff["added"].items()
    )
    lines.extend(
        f"- Removed: `{col}` ({dtype})" for col, dtype in schema_diff["removed"].items()
    )
    lines.extend(
        f"- Dtype changed: `{col}` {old} -> {new}"
        for col, (old, new) in schema_diff["dtype_changed"].items()
    )
    return "\n".join(lines)


def _generate_str_churn_section(churn: dict | None, max_removed_share: float) -> str:
    """Render UPRN churn and any above-tolerance warning as a markdown section."""
    lines = ["## UPRN churn", ""]
    if churn is None:
        lines.append("Skipped: no UPRN column in this stage's output.")
        return "\n".join(lines)
    lines.extend(
        [
            "| Added | Removed | Retained |",
            "| --- | --- | --- |",
            f"| {churn['n_added']} | {churn['n_removed']} | {churn['n_retained']} |",
            "",
            (
                f"{churn['removed_share']:.1%} of old UPRNs were removed "
                f"(rubric tolerance: {max_removed_share:.1%})."
            ),
        ]
    )
    churn_note = generate_str_churn_note(churn, max_removed_share)
    if churn_note:
        lines.extend(["", churn_note])
    return "\n".join(lines)


def _generate_str_transitions_section(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> str:
    """Render the UPRN-level tech transition matrix as a markdown section."""
    lines = ["## Tech-assignment transitions (UPRN-level)", ""]
    transitions = generate_df_tech_transitions(df_old, df_new)
    if transitions is None:
        lines.append(
            f"Skipped: `{TECH_COL}` column missing from one or both versions "
            "(see schema diff)."
        )
        return "\n".join(lines)
    if transitions.is_empty():
        lines.append("No UPRNs retained across versions; matrix skipped.")
        return "\n".join(lines)
    # Pivot the index under an internal name: a real tech label equal to
    # "assigned_tech_old" would otherwise collide with the pivoted index
    # column and crash the pivot.
    matrix = (
        transitions.rename({"assigned_tech_old": "_old_tech"})
        .pivot(on="assigned_tech_new", index="_old_tech", values="n_uprns")
        .fill_null(0)
    )
    new_techs = sorted(col for col in matrix.columns if col != "_old_tech")
    lines.extend(
        [
            "| Old tech \\ New tech | " + " | ".join(new_techs) + " |",
            "| --- |" + " --- |" * len(new_techs),
        ]
    )
    for row in matrix.sort("_old_tech").iter_rows(named=True):
        cells = " | ".join(str(row[tech]) for tech in new_techs)
        lines.append(f"| {row['_old_tech']} | {cells} |")
    return "\n".join(lines)


def _generate_str_input_changes_section(manifest_old: dict, manifest_new: dict) -> str:
    """Render the manifest-recorded input version changes as a markdown section."""
    changes = generate_dict_input_version_changes(manifest_old, manifest_new)
    lines = ["## Input version changes", ""]
    if not any(changes.values()):
        lines.append("No recorded input version changes.")
    lines.extend(
        f"- Changed: `{key}` {old} -> {new}"
        for key, (old, new) in changes["changed"].items()
    )
    lines.extend(f"- Added: `{key}` {path}" for key, path in changes["added"].items())
    lines.extend(
        f"- Removed: `{key}` {path}" for key, path in changes["removed"].items()
    )
    return "\n".join(lines)


def _generate_str_commit_log_section(
    manifest_old: dict, manifest_new: dict, stage: str
) -> str:
    """Render the module-scoped commit log between recorded commits."""
    commit_old = manifest_old["git_commit"]
    commit_new = manifest_new["git_commit"]
    span = f"`{commit_old[:7]}..{commit_new[:7]}`"
    lines = ["## Module-scoped commit log", ""]
    commits = generate_list_commit_log(commit_old, commit_new, stage)
    if commits is None:
        lines.append(
            "Commit log unavailable: a recorded commit is unknown or not in "
            "local git history (try fetching first)."
        )
    elif commit_old == commit_new:
        lines.append(f"Both versions were produced by the same commit `{commit_old}`.")
    elif not commits:
        lines.append(f"No commits touched this stage's modules in {span}.")
    else:
        lines.append(f"Commits touching this stage's modules in {span}:")
        lines.append("")
        lines.extend(f"- {commit}" for commit in commits)
    return "\n".join(lines)


def generate_str_report(
    df_old: pl.DataFrame,
    df_new: pl.DataFrame,
    manifest_old: dict | None,
    manifest_new: dict | None,
    stage: str,
    local_authority: str,
    trigger: str,
    release_date_old: str,
    release_date_new: str,
    path_old: str,
    path_new: str,
) -> str:
    """
    Assemble the full markdown comparison report.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output
        manifest_old: older version's run manifest, or None when missing
        manifest_new: newer version's run manifest, or None when missing
        stage: pipeline stage the outputs belong to
        local_authority: local authority slug the outputs cover
        trigger: rubric the comparison is read against
        release_date_old: dated version folder of the older output
        release_date_new: dated version folder of the newer output
        path_old: S3 path of the older output
        path_new: S3 path of the newer output

    Returns:
        str: markdown report; lineage sections are replaced by a note when a
            version's run manifest is missing (outputs predating manifests)
    """
    tolerances = get_dict_tolerances(trigger)
    sections = [
        f"# Cross-version comparison: {stage} — {local_authority}",
        "\n".join(
            [
                f"- Old version: {release_date_old} (`{path_old}`)",
                f"- New version: {release_date_new} (`{path_new}`)",
                f"- Trigger: `{trigger}` — read against this rubric's tolerances",
                "- Generated: "
                + datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ]
        ),
        _generate_str_counts_section(generate_dict_count_delta(df_old, df_new)),
        _generate_str_schema_section(generate_dict_schema_diff(df_old, df_new)),
        _generate_str_churn_section(
            generate_dict_uprn_churn(df_old, df_new),
            tolerances["max_removed_uprn_share"],
        ),
    ]
    if stage == "decision_tree":
        sections.append(_generate_str_transitions_section(df_old, df_new))
    if manifest_old is None or manifest_new is None:
        missing_versions = [
            label
            for label, manifest in (("old", manifest_old), ("new", manifest_new))
            if manifest is None
        ]
        missing = " and ".join(missing_versions)
        noun = "version" if len(missing_versions) == 1 else "versions"
        sections.append(
            f"## Lineage\n\nRun manifest missing for the {missing} {noun} "
            "(output predates run manifests); input-version and commit-log "
            "sections skipped."
        )
    else:
        sections.append(_generate_str_input_changes_section(manifest_old, manifest_new))
        sections.append(
            _generate_str_commit_log_section(manifest_old, manifest_new, stage)
        )
    return "\n\n".join(sections) + "\n"


def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Compare two dated versions of a pipeline stage output."
    )
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_OUTPUT_DATASETS))
    parser.add_argument(
        "--local_authority",
        required=True,
        help="Local authority slug as used in output paths, e.g. 'plymouth'.",
    )
    parser.add_argument(
        "--old_release_date", required=True, help="Older version, YYYYMMDD."
    )
    parser.add_argument(
        "--new_release_date", required=True, help="Newer version, YYYYMMDD."
    )
    parser.add_argument(
        "--trigger",
        required=True,
        choices=sorted(TOLERANCES),
        help="Why the comparison is run; picks the tolerance rubric.",
    )
    parser.add_argument(
        "--report_dir",
        default=str(PROJECT_DIR / "outputs" / "comparisons"),
        help="Directory the markdown report is written to.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Run the comparison, write the markdown report and log a console summary.

    Args:
        args: parsed CLI arguments
    """
    release_date_old = save_utils.get_str_release_date(args.old_release_date)
    release_date_new = save_utils.get_str_release_date(args.new_release_date)
    path_old = get_str_stage_output_path(
        args.stage, args.local_authority, release_date_old, check_exists=True
    )
    path_new = get_str_stage_output_path(
        args.stage, args.local_authority, release_date_new, check_exists=True
    )
    df_old = load_transform_df_stage_output(path_old)
    df_new = load_transform_df_stage_output(path_new)
    report = generate_str_report(
        df_old,
        df_new,
        load_dict_manifest(path_old),
        load_dict_manifest(path_new),
        stage=args.stage,
        local_authority=args.local_authority,
        trigger=args.trigger,
        release_date_old=release_date_old,
        release_date_new=release_date_new,
        path_old=path_old,
        path_new=path_new,
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"{args.stage}_{args.local_authority}_"
        f"{release_date_old}_vs_{release_date_new}.md"
    )
    report_path.write_text(report)

    counts = generate_dict_count_delta(df_old, df_new)
    logging.info(
        "Rows: %s -> %s (%+d)",
        counts["rows_old"],
        counts["rows_new"],
        counts["rows_delta"],
    )
    churn = generate_dict_uprn_churn(df_old, df_new)
    if churn:
        logging.info(
            "UPRN churn: %d added, %d removed, %d retained",
            churn["n_added"],
            churn["n_removed"],
            churn["n_retained"],
        )
        churn_note = generate_str_churn_note(
            churn, get_dict_tolerances(args.trigger)["max_removed_uprn_share"]
        )
        if churn_note:
            logging.warning(churn_note)
    logging.info("Report written to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_arguments())
