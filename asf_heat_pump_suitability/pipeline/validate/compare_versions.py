"""
Compare two dated versions of a pipeline stage output for one local authority.

Reports row/UPRN count deltas, schema diff, UPRN churn, per-tech counts and
the tech-assignment transition matrix (decision-tree stage), and a
module-scoped commit log between the two versions' recorded commits. With a
--trigger, checks are read against that rubric's tolerances; without one, the
report presents raw numbers only. Writes a local markdown report and logs a
console summary.

Usage (compare the latest two versions, raw numbers only):
python -m asf_heat_pump_suitability.pipeline.validate.compare_versions \
    --stage decision_tree \
    --local_authority plymouth

Usage (explicit versions, read against a rubric):
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
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import polars as pl
import pyarrow.parquet as pq
import s3fs

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import geo_utils, manifest_utils, save_utils

UPRN_COL = "UPRN"
TECH_COL = "assigned_tech"
NULL_TECH_LABEL = "(null)"

STAGE_MODULE_PATHS = config["compare_versions"]["stage_module_paths"]
STAGE_OUTPUT_DATASETS = config["compare_versions"]["stage_output_datasets"]
BUILDINGS_DATASET = config["compare_versions"]["decision_tree_buildings_dataset"]
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


def _expr_tech_canonical() -> pl.Expr:
    """
    The tech-assignment column with nulls as a regular report label, so the
    counts table and the transition matrix cannot diverge on how a null
    tech is presented.
    """
    return pl.col(TECH_COL).fill_null(NULL_TECH_LABEL)


def generate_dict_uprn_churn(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict | None:
    """
    Count UPRNs added, removed and retained between two versions of an output.

    UPRNs are compared as canonical strings so a dtype change between
    versions does not read as full churn. Null UPRNs are excluded from the
    churn sets (any number of them would collapse to one set element and
    silently undercount) and reported as their own counts instead.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: added/removed/retained counts, the removed and added shares
            (each as a fraction of the old version's UPRN count), and
            per-version null-UPRN counts, or None when either version has
            no UPRN column
    """
    if UPRN_COL not in df_old.columns or UPRN_COL not in df_new.columns:
        return None
    # Null UPRNs are counted separately below; drop them here so they cannot
    # hide in the sets (any number of nulls would collapse to one element).
    uprns_old = set(
        df_old.drop_nulls(UPRN_COL).select(_expr_uprn_canonical()).to_series().to_list()
    )
    uprns_new = set(
        df_new.drop_nulls(UPRN_COL).select(_expr_uprn_canonical()).to_series().to_list()
    )
    n_added = len(uprns_new - uprns_old)
    n_removed = len(uprns_old - uprns_new)
    n_old = len(uprns_old)
    return {
        "n_added": n_added,
        "n_removed": n_removed,
        "n_retained": len(uprns_old & uprns_new),
        "removed_share": n_removed / n_old if n_old else 0.0,
        "added_share": n_added / n_old if n_old else 0.0,
        "n_null_old": df_old[UPRN_COL].null_count(),
        "n_null_new": df_new[UPRN_COL].null_count(),
    }


def generate_str_churn_note(
    share: float, max_share: float, description: str
) -> str | None:
    """
    Warn when a churn share is above its tolerance.

    Args:
        share: observed share, as a fraction of the old version's UPRN count
        max_share: the tolerance for this share, from the trigger's settings
        description: what the share measures, e.g. "of old UPRNs were removed"

    Returns:
        str: warning naming the share and the tolerance, or None when the
            share is within tolerance
    """
    if share <= max_share:
        return None
    return (
        f"WARNING: {share:.1%} {description}, above the "
        f"{max_share:.1%} tolerance for this trigger."
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
        _expr_tech_canonical().alias("assigned_tech_old"),
    ).unique(subset=[UPRN_COL], keep="first")
    new = df_new.select(
        _expr_uprn_canonical(),
        _expr_tech_canonical().alias("assigned_tech_new"),
    ).unique(subset=[UPRN_COL], keep="first")
    return (
        old.join(new, on=UPRN_COL, how="inner")
        .group_by("assigned_tech_old", "assigned_tech_new")
        .agg(pl.len().alias("n_uprns"))
        .sort("n_uprns", "assigned_tech_old", "assigned_tech_new", descending=True)
    )


def generate_df_tech_counts(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> pl.DataFrame | None:
    """
    Count rows per tech assignment in each version of an output.

    Unlike the transition matrix these are per-version tallies needing no
    cross-version key, so they also apply to the building-level output,
    whose IDs are not stable across versions. Null tech assignments are
    labelled "(null)"; a tech present in only one version counts 0 in the
    other.

    Args:
        df_old: older version of a decision-tree output
        df_new: newer version of a decision-tree output

    Returns:
        pl.DataFrame: one row per tech with n_old, n_new and n_delta counts,
            sorted by tech, or None when either version has no
            tech-assignment column
    """
    if TECH_COL not in df_old.columns or TECH_COL not in df_new.columns:
        return None

    def tally(df: pl.DataFrame, alias: str) -> pl.DataFrame:
        return df.group_by(_expr_tech_canonical()).agg(
            pl.len().cast(pl.Int64).alias(alias)
        )

    return (
        tally(df=df_old, alias="n_old")
        .join(tally(df=df_new, alias="n_new"), on=TECH_COL, how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("n_new") - pl.col("n_old")).alias("n_delta"))
        .sort(TECH_COL)
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
    ancestor_check = manifest_utils.run_git_or_none(
        ["git", "merge-base", "--is-ancestor", commit_old, commit_new],
        "%s is not an ancestor of %s (or a commit is unfetched); "
        "old..new would silently omit commits, so the log is skipped.",
        commit_old,
        commit_new,
    )
    if ancestor_check is None:
        return None
    result = manifest_utils.run_git_or_none(
        [
            "git",
            "log",
            "--oneline",
            f"{commit_old}..{commit_new}",
            "--",
            *STAGE_MODULE_PATHS[stage],
        ],
        "git log %s..%s failed; are both commits fetched locally?",
        commit_old,
        commit_new,
    )
    if result is None:
        return None
    return result.stdout.splitlines()


def _get_str_output_path(
    dataset: str, local_authority: str, release_date: str, check_exists: bool
) -> str:
    """Build an output dataset's exact dated S3 path (single site for the
    template format kwargs; save_utils validates the date)."""
    return save_utils.get_str_output_path(
        dataset,
        release_date=release_date,
        check_exists=check_exists,
        local_authority=local_authority,
        local_authorities=local_authority,
        tolerance_m=config["constant"]["clustering"]["tolerance_m"],
    )


def _generate_str_output_glob(
    dataset: str, local_authority: str, release_date: str = "*"
) -> str:
    """Build an S3 glob over an output dataset's path template, wildcarding
    the dated directory (unless given) and the clustering tolerance — so
    versions saved under a previous tolerance stay discoverable."""
    return config["output"]["dataset"][dataset].format(
        release_date=release_date,
        local_authority=local_authority,
        local_authorities=local_authority,
        tolerance_m="*",
    )


def get_str_stage_output_path(
    stage: str,
    local_authority: str,
    release_date: str,
    check_exists: bool = False,
) -> str:
    """
    Build the S3 path of the output a stage's comparison reads.

    The contextual-features filename embeds the clustering tolerance; when
    `check_exists` finds nothing under the current config tolerance, a
    version saved under a previous tolerance is resolved by glob, so past
    releases stay comparable across tolerance changes.

    Args:
        stage: pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority: local authority slug used in output paths
        release_date: dated version folder in YYYYMMDD format
        check_exists: if True, raise when no file exists at the path

    Returns:
        str: S3 path of the stage's output for that version
    """
    dataset = STAGE_OUTPUT_DATASETS[stage]
    try:
        return _get_str_output_path(
            dataset=dataset,
            local_authority=local_authority,
            release_date=release_date,
            check_exists=check_exists,
        )
    except FileNotFoundError:
        matches = s3fs.S3FileSystem().glob(
            _generate_str_output_glob(
                dataset=dataset,
                local_authority=local_authority,
                release_date=release_date,
            )
        )
        if len(matches) != 1:
            raise
        path = f"s3://{matches[0]}"
        logging.info(
            "No %s output under the current clustering tolerance; resolved %s",
            stage,
            path,
        )
        return path


def generate_list_release_dates(stage: str, local_authority: str) -> list[str]:
    """
    List the dated versions of a stage's output available on S3.

    Args:
        stage: pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority: local authority slug used in output paths

    Returns:
        list[str]: distinct release dates in YYYYMMDD format, oldest first
    """
    pattern = _generate_str_output_glob(
        dataset=STAGE_OUTPUT_DATASETS[stage], local_authority=local_authority
    )
    release_dates = set()
    for path in s3fs.S3FileSystem().glob(pattern):
        # The release date is the output file's parent directory in every
        # output path template.
        segment = path.rsplit("/", 2)[-2]
        try:
            release_dates.add(save_utils.get_str_release_date(segment))
        except ValueError:
            continue  # not a dated version directory
    return sorted(release_dates)


def get_tuple_default_release_dates(
    stage: str, local_authority: str
) -> tuple[str, str]:
    """
    Pick the latest two dated versions of a stage's output to compare.

    Args:
        stage: pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority: local authority slug used in output paths

    Returns:
        tuple[str, str]: (older, newer) of the two latest release dates

    Raises:
        FileNotFoundError: when fewer than two dated versions exist
    """
    release_dates = generate_list_release_dates(stage, local_authority)
    if len(release_dates) < 2:
        raise FileNotFoundError(
            f"Found {len(release_dates)} dated version(s) of {stage} for "
            f"{local_authority} on S3 ({release_dates or 'none'}); need two to "
            "compare. Pass --old_release_date and --new_release_date explicitly."
        )
    return release_dates[-2], release_dates[-1]


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
            # Not every geojson is saved in EPSG:4326, so check rather than
            # assume. The comparison only reads the non-geometry columns, so
            # this is protection for future geometry checks, not this report.
            gdf = geo_utils.verify_gdf_crs(gdf, target_crs="EPSG:4326")
        except ValueError:
            logging.warning("No features in geojson at %s; comparing as empty.", path)
            return pl.DataFrame()
        return pl.from_pandas(gdf.drop(columns="geometry"))
    raise ValueError(f"Cannot compare file type of {path}; expected parquet/geojson.")


def load_df_buildings_tech(path: str) -> pl.DataFrame:
    """
    Load only the tech-assignment column of a building-level output.

    The building-level output feeds the per-tech counts alone, so a single
    column is fetched rather than the full parquet. A version without the
    column loads as an empty frame, which the counts section reports as a
    missing column.

    Args:
        path: S3 path of the building-level output parquet

    Returns:
        pl.DataFrame: the tech-assignment column, or an empty frame
    """
    if TECH_COL not in pq.read_schema(path).names:
        return pl.DataFrame()
    return pl.from_arrow(pq.read_table(path, columns=[TECH_COL]))


def load_tuple_df_buildings(
    local_authority: str, release_date_old: str, release_date_new: str
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """
    Load the tech column of both building-level decision-tree outputs.

    Both paths are existence-checked before anything is downloaded. Any
    missing or unreadable output degrades to (None, None) with a logged
    warning naming the cause, so the building-level counts section becomes
    a note instead of aborting the whole comparison.

    Args:
        local_authority: local authority slug used in output paths
        release_date_old: dated version folder of the older output
        release_date_new: dated version folder of the newer output

    Returns:
        tuple: (older, newer) building-level tech columns, or (None, None)
    """
    try:
        # The building-level output is read only for the per-tech counts;
        # every other decision-tree check uses the UPRN-level output.
        path_old = _get_str_output_path(
            dataset=BUILDINGS_DATASET,
            local_authority=local_authority,
            release_date=release_date_old,
            check_exists=True,
        )
        path_new = _get_str_output_path(
            dataset=BUILDINGS_DATASET,
            local_authority=local_authority,
            release_date=release_date_new,
            check_exists=True,
        )
        return load_df_buildings_tech(path=path_old), load_df_buildings_tech(
            path=path_new
        )
    except (OSError, ValueError) as error:
        logging.warning(
            "Building-level output unavailable (%s); its per-tech counts "
            "section is skipped.",
            error,
        )
        return None, None


def _render_section(title: str, *body: str) -> str:
    """Render a markdown section: a `## Title` heading, then body lines."""
    return "\n".join([f"## {title}", "", *body])


def _generate_str_counts_section(counts: dict) -> str:
    """Render the row/UPRN count delta as a markdown section."""
    lines = [
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
    return _render_section("Row and UPRN counts", *lines)


def _generate_str_schema_section(schema_diff: dict) -> str:
    """Render the schema diff as a markdown section."""
    lines = []
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
    return _render_section("Schema diff", *lines)


def _generate_str_churn_section(churn: dict | None, tolerances: dict | None) -> str:
    """
    Render UPRN churn as a markdown section.

    Args:
        churn: churn counts and shares, as returned by
            `generate_dict_uprn_churn`, or None when the stage has no UPRN column
        tolerances: the trigger's tolerance settings, or None when the
            comparison was run without a trigger (shares are then shown
            without warnings)
    """
    if churn is None:
        return _render_section(
            "UPRN churn", "Skipped: no UPRN column in this stage's output."
        )
    if tolerances is not None:
        removed_suffix = f"(tolerance: {tolerances['max_removed_uprn_share']:.1%})."
        added_suffix = f"(tolerance: {tolerances['max_added_uprn_share']:.1%})."
    else:
        removed_suffix = added_suffix = (
            "(no trigger supplied; not checked against a tolerance)."
        )
    lines = [
        "| Added | Removed | Retained |",
        "| --- | --- | --- |",
        f"| {churn['n_added']} | {churn['n_removed']} | {churn['n_retained']} |",
        "",
        f"{churn['removed_share']:.1%} of old UPRNs were removed {removed_suffix}",
        (
            f"New UPRNs equal to {churn['added_share']:.1%} of the old version "
            f"were added {added_suffix}"
        ),
    ]
    if churn["n_null_old"] or churn["n_null_new"]:
        lines.extend(
            [
                "",
                (
                    f"{churn['n_null_old']} null UPRNs in the old version and "
                    f"{churn['n_null_new']} in the new were excluded from churn."
                ),
            ]
        )
    if tolerances is not None:
        for note in _generate_list_churn_notes(churn=churn, tolerances=tolerances):
            lines.extend(["", note])
    return _render_section("UPRN churn", *lines)


def _generate_list_churn_notes(churn: dict, tolerances: dict) -> list[str]:
    """Collect the above-tolerance warnings for removed and added UPRN shares."""
    notes = [
        generate_str_churn_note(
            share=churn["removed_share"],
            max_share=tolerances["max_removed_uprn_share"],
            description="of old UPRNs were removed",
        ),
        generate_str_churn_note(
            share=churn["added_share"],
            max_share=tolerances["max_added_uprn_share"],
            description="of the old UPRN count was added as new UPRNs",
        ),
    ]
    return [note for note in notes if note is not None]


def _generate_str_tech_counts_section(
    df_old: pl.DataFrame | None, df_new: pl.DataFrame | None, level: str
) -> str:
    """Render per-tech counts for one output level as a markdown section."""
    title = f"Per-tech counts ({level})"
    if df_old is None or df_new is None:
        return _render_section(
            title, "Skipped: output missing for one or both versions."
        )
    counts = generate_df_tech_counts(df_old=df_old, df_new=df_new)
    if counts is None:
        return _render_section(
            title,
            f"Skipped: `{TECH_COL}` column missing from one or both versions "
            "(see schema diff).",
        )
    lines = [
        "| Tech | Old | New | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for row in counts.iter_rows(named=True):
        lines.append(
            f"| {row[TECH_COL]} | {row['n_old']} | {row['n_new']} "
            f"| {row['n_delta']:+d} |"
        )
    return _render_section(title, *lines)


def _generate_str_transitions_section(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> str:
    """Render the UPRN-level tech transition matrix as a markdown section."""
    title = "Tech-assignment transitions (UPRN-level)"
    transitions = generate_df_tech_transitions(df_old=df_old, df_new=df_new)
    if transitions is None:
        return _render_section(
            title,
            f"Skipped: `{TECH_COL}` column missing from one or both versions "
            "(see schema diff).",
        )
    if transitions.is_empty():
        return _render_section(
            title, "No UPRNs retained across versions; matrix skipped."
        )
    # Pivot the index under an internal name: a real tech label equal to
    # "assigned_tech_old" would otherwise collide with the pivoted index
    # column and crash the pivot.
    matrix = (
        transitions.rename({"assigned_tech_old": "_old_tech"})
        .pivot(on="assigned_tech_new", index="_old_tech", values="n_uprns")
        .fill_null(0)
    )
    new_techs = sorted(col for col in matrix.columns if col != "_old_tech")
    lines = [
        "| Old tech \\ New tech | " + " | ".join(new_techs) + " |",
        "| --- |" + " --- |" * len(new_techs),
    ]
    for row in matrix.sort("_old_tech").iter_rows(named=True):
        cells = " | ".join(str(row[tech]) for tech in new_techs)
        lines.append(f"| {row['_old_tech']} | {cells} |")
    return _render_section(title, *lines)


def _generate_str_input_changes_section(manifest_old: dict, manifest_new: dict) -> str:
    """Render the manifest-recorded input version changes as a markdown section."""
    changes = generate_dict_input_version_changes(
        manifest_old=manifest_old, manifest_new=manifest_new
    )
    lines = []
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
    return _render_section("Input version changes", *lines)


def _generate_str_commit_log_section(
    manifest_old: dict, manifest_new: dict, stage: str
) -> str:
    """Render the module-scoped commit log between recorded commits."""
    commit_old = manifest_old["git_commit"]
    commit_new = manifest_new["git_commit"]
    span = f"`{commit_old[:7]}..{commit_new[:7]}`"
    commits = generate_list_commit_log(
        commit_old=commit_old, commit_new=commit_new, stage=stage
    )
    if commits is None:
        lines = [
            "Commit log unavailable: a recorded commit is unknown or not in "
            "local git history (try fetching first)."
        ]
    elif commit_old == commit_new:
        lines = [f"Both versions were produced by the same commit `{commit_old}`."]
    elif not commits:
        lines = [f"No commits touched this stage's modules in {span}."]
    else:
        lines = [f"Commits touching this stage's modules in {span}:", ""]
        lines.extend(f"- {commit}" for commit in commits)
    return _render_section("Module-scoped commit log", *lines)


def generate_str_report(
    df_old: pl.DataFrame,
    df_new: pl.DataFrame,
    manifest_old: dict | None,
    manifest_new: dict | None,
    stage: str,
    local_authority: str,
    trigger: str | None,
    release_date_old: str,
    release_date_new: str,
    path_old: str,
    path_new: str,
    df_buildings_old: pl.DataFrame | None = None,
    df_buildings_new: pl.DataFrame | None = None,
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
        trigger: rubric the comparison is read against, or None for raw
            numbers with no rubric interpretation or tolerance warnings
        release_date_old: dated version folder of the older output
        release_date_new: dated version folder of the newer output
        path_old: S3 path of the older output
        path_new: S3 path of the newer output
        df_buildings_old: older building-level decision-tree output, or None
            when missing (its per-tech counts section is then skipped)
        df_buildings_new: newer building-level output, or None when missing

    Returns:
        str: markdown report; lineage sections are replaced by a note when a
            version's run manifest is missing (outputs predating manifests)
    """
    tolerances = get_dict_tolerances(trigger) if trigger is not None else None
    trigger_line = (
        f"- Trigger: `{trigger}` — read against this rubric's tolerances"
        if trigger is not None
        else "- Trigger: not supplied — raw numbers only, no rubric interpretation"
    )
    sections = [
        f"# Cross-version comparison: {stage} — {local_authority}",
        "\n".join(
            [
                f"- Old version: {release_date_old} (`{path_old}`)",
                f"- New version: {release_date_new} (`{path_new}`)",
                trigger_line,
                "- Generated: "
                + datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ]
        ),
        _generate_str_counts_section(
            counts=generate_dict_count_delta(df_old=df_old, df_new=df_new)
        ),
        _generate_str_schema_section(
            schema_diff=generate_dict_schema_diff(df_old=df_old, df_new=df_new)
        ),
        _generate_str_churn_section(
            churn=generate_dict_uprn_churn(df_old=df_old, df_new=df_new),
            tolerances=tolerances,
        ),
    ]
    if stage == "decision_tree":
        sections.append(
            _generate_str_tech_counts_section(
                df_old=df_old, df_new=df_new, level="UPRN-level"
            )
        )
        sections.append(
            _generate_str_tech_counts_section(
                df_old=df_buildings_old,
                df_new=df_buildings_new,
                level="building-level",
            )
        )
        sections.append(_generate_str_transitions_section(df_old=df_old, df_new=df_new))
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
        sections.append(
            _generate_str_input_changes_section(
                manifest_old=manifest_old, manifest_new=manifest_new
            )
        )
        sections.append(
            _generate_str_commit_log_section(
                manifest_old=manifest_old, manifest_new=manifest_new, stage=stage
            )
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
        "--old_release_date",
        help="Older version, YYYYMMDD. Omit both dates to compare the "
        "latest two versions found on S3.",
    )
    parser.add_argument("--new_release_date", help="Newer version, YYYYMMDD.")
    parser.add_argument(
        "--trigger",
        choices=sorted(TOLERANCES),
        help="Why the comparison is run; picks the tolerance rubric. "
        "Omitted: raw numbers only, no rubric interpretation.",
    )
    parser.add_argument(
        "--report_dir",
        default=str(PROJECT_DIR / "outputs" / "comparisons"),
        help="Directory the markdown report is written to.",
    )
    args = parser.parse_args()
    if (args.old_release_date is None) != (args.new_release_date is None):
        parser.error(
            "Pass both --old_release_date and --new_release_date, or neither "
            "to compare the latest two versions."
        )
    return args


if __name__ == "__main__":
    # Orchestration lives in this block, not a main() function, matching the
    # other pipeline entrypoints: intermediates stay in module globals so
    # `python -i -m ...compare_versions ...` leaves them inspectable.
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    # Output paths use lowercase LA slugs; lowercase like the sibling
    # entrypoints (e.g. add_features) so "Plymouth" finds plymouth's outputs.
    local_authority = args.local_authority.lower()
    if args.old_release_date is None:
        release_date_old, release_date_new = get_tuple_default_release_dates(
            stage=args.stage, local_authority=local_authority
        )
        logging.info(
            "No versions supplied; comparing the latest two: %s vs %s",
            release_date_old,
            release_date_new,
        )
    else:
        release_date_old = save_utils.get_str_release_date(args.old_release_date)
        release_date_new = save_utils.get_str_release_date(args.new_release_date)
    path_old = get_str_stage_output_path(
        stage=args.stage,
        local_authority=local_authority,
        release_date=release_date_old,
        check_exists=True,
    )
    path_new = get_str_stage_output_path(
        stage=args.stage,
        local_authority=local_authority,
        release_date=release_date_new,
        check_exists=True,
    )
    df_old = load_transform_df_stage_output(path_old)
    df_new = load_transform_df_stage_output(path_new)
    df_buildings_old = df_buildings_new = None
    if args.stage == "decision_tree":
        df_buildings_old, df_buildings_new = load_tuple_df_buildings(
            local_authority=local_authority,
            release_date_old=release_date_old,
            release_date_new=release_date_new,
        )
    report = generate_str_report(
        df_old=df_old,
        df_new=df_new,
        manifest_old=load_dict_manifest(output_path=path_old),
        manifest_new=load_dict_manifest(output_path=path_new),
        stage=args.stage,
        local_authority=local_authority,
        trigger=args.trigger,
        release_date_old=release_date_old,
        release_date_new=release_date_new,
        path_old=path_old,
        path_new=path_new,
        df_buildings_old=df_buildings_old,
        df_buildings_new=df_buildings_new,
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"{args.stage}_{local_authority}_{release_date_old}_vs_{release_date_new}.md"
    )
    report_path.write_text(report)

    counts = generate_dict_count_delta(df_old=df_old, df_new=df_new)
    logging.info(
        "Rows: %s -> %s (diff: %+d)",
        counts["rows_old"],
        counts["rows_new"],
        counts["rows_delta"],
    )
    churn = generate_dict_uprn_churn(df_old=df_old, df_new=df_new)
    if churn:
        logging.info(
            "UPRN churn: %d added, %d removed, %d retained",
            churn["n_added"],
            churn["n_removed"],
            churn["n_retained"],
        )
        if args.trigger is not None:
            for churn_note in _generate_list_churn_notes(
                churn=churn, tolerances=get_dict_tolerances(trigger=args.trigger)
            ):
                logging.warning(churn_note)
    logging.info("Report written to %s", report_path)
