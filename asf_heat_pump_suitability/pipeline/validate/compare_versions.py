"""
Compare two dated versions of a pipeline stage output for one local authority.

Reports row/UPRN count deltas, schema diff, UPRN churn, per-tech counts and
the tech-assignment transition matrix (decision-tree stage), cluster
count/area deltas and distribution comparisons with overlaid plots (cluster
and contextual-features stages), and a module-scoped commit log between the
two versions' recorded commits. With a --trigger, checks are read against
that rubric's tolerances; without one, the report presents raw numbers only.
Writes a local markdown report (plus distribution plot PNGs next to it) and
logs a console summary.

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
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import s3fs

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import geo_utils, manifest_utils, save_utils

UPRN_COL = "UPRN"
TECH_COL = "assigned_tech"
NULL_TECH_LABEL = "(null)"
CLUSTER_ID_COL = "cluster_id"
AREA_COL = "area_m2"
# Multi-layer front-end outputs (August 2026 onwards) tag every row with
# the layer it belongs to; earlier outputs have no such column.
LAYER_COL = "layer"
# Stages whose outputs carry cluster geometry, and so get the cluster
# count, area and distribution sections.
GEOMETRY_STAGES = ("cluster", "compute_contextual_features")

CLUSTER_LAYER = config["compare_versions"]["cluster_layer"]
STAGE_MODULE_PATHS = config["compare_versions"]["stage_module_paths"]
STAGE_OUTPUT_DATASETS = config["compare_versions"]["stage_output_datasets"]
BUILDINGS_DATASET = config["compare_versions"]["decision_tree_buildings_dataset"]
DISTRIBUTION_COLUMNS = config["compare_versions"]["distribution_columns"]
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


def generate_dict_cluster_count_delta(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> dict | None:
    """
    Compare distinct-cluster counts between two versions of an output.

    Args:
        df_old: older version of a cluster-bearing stage output
        df_new: newer version of a cluster-bearing stage output

    Returns:
        dict: old/new/delta distinct-cluster counts, or None when either
            version has no cluster-id column
    """
    if CLUSTER_ID_COL not in df_old.columns or CLUSTER_ID_COL not in df_new.columns:
        return None
    n_old = df_old[CLUSTER_ID_COL].n_unique()
    n_new = df_new[CLUSTER_ID_COL].n_unique()
    return {
        "clusters_old": n_old,
        "clusters_new": n_new,
        "clusters_delta": n_new - n_old,
    }


def filter_df_clusters_layer(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep only a frame's clusters-layer rows.

    Multi-layer front-end outputs bundle non-cluster layers (ward
    boundaries, anchor loads) with the clusters in one file; the cluster
    checks must not aggregate over them. A frame without a `layer` column
    (a pre-layers output) is all clusters and passes through unchanged.

    Args:
        df: one version of a geometry-stage output (tabular or per-row areas)

    Returns:
        pl.DataFrame: the rows whose layer is the configured clusters layer,
            or the frame itself when it has no `layer` column
    """
    if LAYER_COL not in df.columns:
        return df
    return df.filter(pl.col(LAYER_COL) == CLUSTER_LAYER)


def generate_dict_total_area_delta(
    df_areas_old: pl.DataFrame, df_areas_new: pl.DataFrame
) -> dict:
    """
    Compare total cluster area between two versions, in m² (EPSG:27700).

    Args:
        df_areas_old: older version's per-cluster areas
        df_areas_new: newer version's per-cluster areas

    Returns:
        dict: old/new/delta total areas; a version with no clusters totals 0
    """
    total_old = df_areas_old[AREA_COL].sum()
    total_new = df_areas_new[AREA_COL].sum()
    return {
        "area_m2_old": total_old,
        "area_m2_new": total_new,
        "area_m2_delta": total_new - total_old,
    }


def generate_dict_distribution_stats(df: pl.DataFrame, column: str) -> dict | None:
    """
    Summarise one column's distribution: both quartiles, min, max and mean.

    Q1 and Q3 are reported separately (linear interpolation) so a shifted
    distribution and a widened one stay distinguishable. Nulls are excluded.

    Args:
        df: one version of a stage output
        column: column to summarise

    Returns:
        dict: min/q1/mean/q3/max, or None when the column is missing or has
            no non-null values
    """
    if column not in df.columns:
        return None
    values = df[column].drop_nulls()
    if values.is_empty():
        return None
    return {
        "min": values.min(),
        "q1": values.quantile(0.25, "linear"),
        "mean": values.mean(),
        "q3": values.quantile(0.75, "linear"),
        "max": values.max(),
    }


def get_dict_distribution_frames(
    stage: str,
    df_old: pl.DataFrame,
    df_new: pl.DataFrame,
    df_areas_old: pl.DataFrame | None,
    df_areas_new: pl.DataFrame | None,
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Map each of a stage's distribution columns to the (old, new) frames
    that carry it.

    The derived cluster area reads the geometry-derived frames (when
    loaded); the stage's configured `DISTRIBUTION_COLUMNS` read the tabular
    outputs. Every frame is filtered to the clusters layer here, so the
    stats tables and the plots fed from this mapping cannot diverge on
    which rows they cover.

    Args:
        stage: pipeline stage the outputs belong to
        df_old: older version of the tabular stage output
        df_new: newer version of the tabular stage output
        df_areas_old: older version's per-row areas, or None
        df_areas_new: newer version's per-row areas, or None

    Returns:
        dict: column name to (old, new) clusters-layer frame pair,
            plot/report order
    """
    frames = {}
    if df_areas_old is not None and df_areas_new is not None:
        frames[AREA_COL] = (
            filter_df_clusters_layer(df_areas_old),
            filter_df_clusters_layer(df_areas_new),
        )
    for column in DISTRIBUTION_COLUMNS.get(stage, []):
        frames[column] = (
            filter_df_clusters_layer(df_old),
            filter_df_clusters_layer(df_new),
        )
    return frames


def plot_distribution_overlay(
    values_old: pl.Series, values_new: pl.Series, label: str, path: Path
) -> None:
    """
    Save an overlaid old-vs-new histogram of one distribution as a PNG.

    Both versions share the same bins so the shapes are comparable.

    Args:
        values_old: older version's values
        values_new: newer version's values
        label: distribution name, used for the x-axis and title
        path: file path the PNG is saved to
    """
    old, new = values_old.to_numpy(), values_new.to_numpy()
    bins = np.histogram_bin_edges(np.concatenate([old, new]), bins=40)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(old, bins=bins, alpha=0.6, label="Old", color="tab:blue")
    ax.hist(new, bins=bins, alpha=0.6, label="New", color="tab:orange")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {label}: old vs new")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_dict_distribution_plots(
    frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    plot_dir: Path,
    file_stem: str,
) -> dict[str, str]:
    """
    Save an overlaid old-vs-new histogram for each of a stage's distributions.

    A distribution with a missing column or no values on either side is
    skipped with a warning — its stats section already notes the gap.

    Args:
        frames: column to (old, new) frame pair, as built by
            `get_dict_distribution_frames`
        plot_dir: directory the PNGs are saved to
        file_stem: filename prefix, shared with the markdown report

    Returns:
        dict: column to saved PNG filename (relative to `plot_dir`, so the
            report next to the PNGs can link them relatively)
    """
    plot_files = {}
    for column, (frame_old, frame_new) in frames.items():
        if column not in frame_old.columns or column not in frame_new.columns:
            logging.warning("Column %s missing from one version; plot skipped.", column)
            continue
        values_old = frame_old[column].drop_nulls()
        values_new = frame_new[column].drop_nulls()
        if values_old.is_empty() or values_new.is_empty():
            logging.warning("No %s values in one version; plot skipped.", column)
            continue
        filename = f"{file_stem}_{column}.png"
        plot_distribution_overlay(values_old, values_new, column, plot_dir / filename)
        plot_files[column] = filename
    return plot_files


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
        dict: added/removed/retained counts, the removed share of old
            UPRNs, and per-version null-UPRN counts, or None when either
            version has no UPRN column
    """
    if UPRN_COL not in df_old.columns or UPRN_COL not in df_new.columns:
        return None
    uprns_old = set(
        df_old.drop_nulls(UPRN_COL).select(_expr_uprn_canonical()).to_series().to_list()
    )
    uprns_new = set(
        df_new.drop_nulls(UPRN_COL).select(_expr_uprn_canonical()).to_series().to_list()
    )
    n_removed = len(uprns_old - uprns_new)
    return {
        "n_added": len(uprns_new - uprns_old),
        "n_removed": n_removed,
        "n_retained": len(uprns_old & uprns_new),
        "removed_share": n_removed / len(uprns_old) if uprns_old else 0.0,
        "n_null_old": df_old[UPRN_COL].null_count(),
        "n_null_new": df_new[UPRN_COL].null_count(),
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
        tally(df_old, "n_old")
        .join(tally(df_new, "n_new"), on=TECH_COL, how="full", coalesce=True)
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
            dataset, local_authority, release_date, check_exists
        )
    except FileNotFoundError:
        matches = s3fs.S3FileSystem().glob(
            _generate_str_output_glob(dataset, local_authority, release_date)
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


def get_str_buildings_output_path(
    local_authority: str,
    release_date: str,
    check_exists: bool = False,
) -> str:
    """
    Build the S3 path of the building-level decision-tree output.

    Read only for per-tech marginal counts; the decision-tree comparison's
    other checks read the UPRN-level output (see `STAGE_OUTPUT_DATASETS`).

    Args:
        local_authority: local authority slug used in output paths
        release_date: dated version folder in YYYYMMDD format
        check_exists: if True, raise when no file exists at the path

    Returns:
        str: S3 path of the building-level output for that version
    """
    return _get_str_output_path(
        BUILDINGS_DATASET, local_authority, release_date, check_exists
    )


def generate_list_release_dates(stage: str, local_authority: str) -> list[str]:
    """
    List the dated versions of a stage's output available on S3.

    Args:
        stage: pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority: local authority slug used in output paths

    Returns:
        list[str]: distinct release dates in YYYYMMDD format, oldest first
    """
    pattern = _generate_str_output_glob(STAGE_OUTPUT_DATASETS[stage], local_authority)
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
        except ValueError:
            logging.warning("No features in geojson at %s; comparing as empty.", path)
            return pl.DataFrame()
        return pl.from_pandas(gdf.drop(columns="geometry"))
    raise ValueError(f"Cannot compare file type of {path}; expected parquet/geojson.")


def load_df_cluster_areas(path: str) -> pl.DataFrame:
    """
    Load a geometry-bearing stage output as per-row areas in m².

    Areas are measured in EPSG:27700 (metres); an output saved in another
    CRS — the contextual-features geojson is EPSG:4326, with simplified
    geometry — is reprojected first. A zero-feature geojson degrades to an
    empty frame. A multi-layer output's `layer` column rides along with the
    areas, so the checks can filter to the clusters layer and the per-layer
    table can tally the rest.

    Args:
        path: S3 path of the stage output (.parquet or .geojson)

    Returns:
        pl.DataFrame: one `area_m2` row per feature row (not deduplicated on
            cluster id, so a duplicated cluster contributes each of its
            rows), plus the source's `layer` column when it has one

    Raises:
        ValueError: for file types the comparison cannot read geometry from
    """
    if path.endswith(".parquet"):
        columns = ["geometry"]
        if LAYER_COL in pq.read_schema(path).names:
            columns.append(LAYER_COL)
        gdf = gpd.read_parquet(path, columns=columns)
    elif path.endswith(".geojson"):
        try:
            gdf = base_getters.load_gdf_from_s3_geojson(path, crs="EPSG:4326")
        except ValueError:
            logging.warning("No features in geojson at %s; no areas to load.", path)
            return pl.DataFrame(schema={AREA_COL: pl.Float64})
    else:
        raise ValueError(f"Cannot read geometry of {path}; expected parquet/geojson.")
    gdf = geo_utils.verify_gdf_crs(gdf)
    areas = {AREA_COL: gdf.area.to_numpy()}
    if LAYER_COL in gdf.columns:
        areas[LAYER_COL] = gdf[LAYER_COL].to_numpy()
    return pl.DataFrame(areas)


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
        path_old = get_str_buildings_output_path(
            local_authority, release_date_old, check_exists=True
        )
        path_new = get_str_buildings_output_path(
            local_authority, release_date_new, check_exists=True
        )
        return load_df_buildings_tech(path_old), load_df_buildings_tech(path_new)
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


def _generate_str_churn_section(
    churn: dict | None, max_removed_share: float | None
) -> str:
    """Render UPRN churn, checked against the rubric tolerance when one is
    supplied (i.e. the comparison was run with a trigger)."""
    if churn is None:
        return _render_section(
            "UPRN churn", "Skipped: no UPRN column in this stage's output."
        )
    share_suffix = (
        f"(rubric tolerance: {max_removed_share:.1%})."
        if max_removed_share is not None
        else "(no trigger supplied; not checked against a tolerance)."
    )
    lines = [
        "| Added | Removed | Retained |",
        "| --- | --- | --- |",
        f"| {churn['n_added']} | {churn['n_removed']} | {churn['n_retained']} |",
        "",
        f"{churn['removed_share']:.1%} of old UPRNs were removed {share_suffix}",
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
    if max_removed_share is not None:
        churn_note = generate_str_churn_note(churn, max_removed_share)
        if churn_note:
            lines.extend(["", churn_note])
    return _render_section("UPRN churn", *lines)


def _format_stat(value: float | int) -> str:
    """Format a statistic for a report table (floats to one decimal place)."""
    return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"


def _generate_str_cluster_geometry_section(
    count_delta: dict | None,
    area_delta: dict | None,
    stage: str,
    layer_filtered: bool = False,
) -> str:
    """Render cluster count and total area deltas as a markdown section,
    with the CRS/units stated, the clusters-layer scope named when a
    multi-layer output was filtered, and the simplified-geometry caveat
    for the contextual-features stage."""
    lines = [
        "| Metric | Old | New | Delta |",
        "| --- | --- | --- | --- |",
    ]
    if count_delta is not None:
        lines.append(
            f"| Clusters | {count_delta['clusters_old']} "
            f"| {count_delta['clusters_new']} "
            f"| {count_delta['clusters_delta']:+d} |"
        )
    if area_delta is not None:
        lines.append(
            f"| Total area (m²) | {area_delta['area_m2_old']:,.1f} "
            f"| {area_delta['area_m2_new']:,.1f} "
            f"| {area_delta['area_m2_delta']:+,.1f} |"
        )
    if layer_filtered:
        lines.extend(
            [
                "",
                "This output bundles multiple front-end layers; the cluster "
                f"checks and distributions cover the `{CLUSTER_LAYER}` layer "
                f"only. A version without a `{LAYER_COL}` column predates "
                "layered outputs and counts entirely as clusters.",
            ]
        )
    if count_delta is None:
        lines.extend(
            [
                "",
                f"Cluster count skipped: no `{CLUSTER_ID_COL}` column in one or "
                "both versions (see schema diff).",
            ]
        )
    if area_delta is None:
        lines.extend(["", "Total area skipped: geometry unavailable."])
    else:
        lines.extend(
            ["", "Areas are computed in EPSG:27700 (British National Grid), in m²."]
        )
        if stage == "compute_contextual_features":
            lines.append(
                "Note: this stage's areas are measured on simplified geometry "
                "(reprojected from EPSG:4326); small differences from the "
                "cluster stage are simplification artefacts, not drift."
            )
    return _render_section("Cluster geometry", *lines)


def _generate_str_distribution_section(
    column: str,
    frame_old: pl.DataFrame,
    frame_new: pl.DataFrame,
    plot_file: str | None,
) -> str:
    """Render one distribution's per-version statistics as a markdown
    section, with its overlaid plot embedded when one was saved."""
    title = f"Distribution: {column}"
    stats_old = generate_dict_distribution_stats(frame_old, column)
    stats_new = generate_dict_distribution_stats(frame_new, column)
    if stats_old is None or stats_new is None:
        return _render_section(
            title, f"Skipped: no `{column}` values in one or both versions."
        )
    lines = [
        "| Statistic | Old | New |",
        "| --- | --- | --- |",
    ]
    labels = {"min": "Min", "q1": "Q1", "mean": "Mean", "q3": "Q3", "max": "Max"}
    for key, label in labels.items():
        lines.append(
            f"| {label} | {_format_stat(stats_old[key])} "
            f"| {_format_stat(stats_new[key])} |"
        )
    if plot_file is not None:
        lines.extend(["", f"![Distribution of {column}: old vs new]({plot_file})"])
    return _render_section(title, *lines)


def _generate_str_tech_counts_section(
    df_old: pl.DataFrame | None, df_new: pl.DataFrame | None, level: str
) -> str:
    """Render per-tech counts for one output level as a markdown section."""
    title = f"Per-tech counts ({level})"
    if df_old is None or df_new is None:
        return _render_section(
            title, "Skipped: output missing for one or both versions."
        )
    counts = generate_df_tech_counts(df_old, df_new)
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
    transitions = generate_df_tech_transitions(df_old, df_new)
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
    changes = generate_dict_input_version_changes(manifest_old, manifest_new)
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
    commits = generate_list_commit_log(commit_old, commit_new, stage)
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
    df_areas_old: pl.DataFrame | None = None,
    df_areas_new: pl.DataFrame | None = None,
    plot_files: dict[str, str] | None = None,
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
        df_areas_old: older version's per-cluster areas (geometry stages), or
            None (the total-area check is then skipped)
        df_areas_new: newer version's per-cluster areas, or None
        plot_files: distribution column to saved plot filename, embedded as
            image links; None embeds no plots

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
        _generate_str_counts_section(generate_dict_count_delta(df_old, df_new)),
        _generate_str_schema_section(generate_dict_schema_diff(df_old, df_new)),
        _generate_str_churn_section(
            generate_dict_uprn_churn(df_old, df_new),
            tolerances["max_removed_uprn_share"] if tolerances is not None else None,
        ),
    ]
    if stage in GEOMETRY_STAGES:
        area_delta = (
            generate_dict_total_area_delta(
                filter_df_clusters_layer(df_areas_old),
                filter_df_clusters_layer(df_areas_new),
            )
            if df_areas_old is not None and df_areas_new is not None
            else None
        )
        layer_filtered = any(
            LAYER_COL in frame.columns
            for frame in (df_old, df_new, df_areas_old, df_areas_new)
            if frame is not None
        )
        sections.append(
            _generate_str_cluster_geometry_section(
                generate_dict_cluster_count_delta(
                    filter_df_clusters_layer(df_old), filter_df_clusters_layer(df_new)
                ),
                area_delta,
                stage,
                layer_filtered,
            )
        )
        frames = get_dict_distribution_frames(
            stage, df_old, df_new, df_areas_old, df_areas_new
        )
        sections.extend(
            _generate_str_distribution_section(
                column, frame_old, frame_new, (plot_files or {}).get(column)
            )
            for column, (frame_old, frame_new) in frames.items()
        )
    if stage == "decision_tree":
        sections.append(_generate_str_tech_counts_section(df_old, df_new, "UPRN-level"))
        sections.append(
            _generate_str_tech_counts_section(
                df_buildings_old, df_buildings_new, "building-level"
            )
        )
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
            args.stage, local_authority
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
        args.stage, local_authority, release_date_old, check_exists=True
    )
    path_new = get_str_stage_output_path(
        args.stage, local_authority, release_date_new, check_exists=True
    )
    df_old = load_transform_df_stage_output(path_old)
    df_new = load_transform_df_stage_output(path_new)
    df_buildings_old = df_buildings_new = None
    if args.stage == "decision_tree":
        df_buildings_old, df_buildings_new = load_tuple_df_buildings(
            local_authority, release_date_old, release_date_new
        )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = (
        f"{args.stage}_{local_authority}_{release_date_old}_vs_{release_date_new}"
    )
    df_areas_old = df_areas_new = plot_files = None
    if args.stage in GEOMETRY_STAGES:
        df_areas_old = load_df_cluster_areas(path_old)
        df_areas_new = load_df_cluster_areas(path_new)
        plot_files = generate_dict_distribution_plots(
            get_dict_distribution_frames(
                args.stage, df_old, df_new, df_areas_old, df_areas_new
            ),
            report_dir,
            report_stem,
        )
    report = generate_str_report(
        df_old,
        df_new,
        load_dict_manifest(path_old),
        load_dict_manifest(path_new),
        stage=args.stage,
        local_authority=local_authority,
        trigger=args.trigger,
        release_date_old=release_date_old,
        release_date_new=release_date_new,
        path_old=path_old,
        path_new=path_new,
        df_buildings_old=df_buildings_old,
        df_buildings_new=df_buildings_new,
        df_areas_old=df_areas_old,
        df_areas_new=df_areas_new,
        plot_files=plot_files,
    )
    report_path = report_dir / f"{report_stem}.md"
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
        if args.trigger is not None:
            churn_note = generate_str_churn_note(
                churn, get_dict_tolerances(args.trigger)["max_removed_uprn_share"]
            )
            if churn_note:
                logging.warning(churn_note)
    logging.info("Report written to %s", report_path)
