"""
Summarise the comparison JSON files of many local authorities for one release.

`pipeline.py --compare` writes one comparison JSON for each LA and stage to
`outputs/data/{la}/{release_date}/comparisons/`. This script reads them for
the given LAs and gives one row for each LA and stage, with a status:

- compared: the JSON compares the release with an older one
- baseline: the JSON has no older release to compare with
- comparison missing: no JSON, but the stage output for the release exists
- stage failed: no JSON and no stage output for the release

Compared rows are sorted by their largest relative change since the older
release. The script writes a parquet table and a markdown summary named for
the LAs they cover, `comparison_summary_{scope}.parquet` and `.md`. Pass
--save to also upload both to `outputs/comparisons/{release_date}/` on S3.

Run it by hand after the Orbit jobs of a release finish.

Usage (all LAs in GB):
python -m asf_heat_pump_suitability.pipeline.validate.summarise_comparisons \
    --release_date 20261014 \
    --local_authorities GB

Usage (named LAs, case insensitive):
python -m asf_heat_pump_suitability.pipeline.validate.summarise_comparisons \
    --release_date 20261014 \
    --local_authorities "East Lothian" "Glasgow City"
"""

import argparse
import json
import logging
from pathlib import Path

import polars as pl
import s3fs

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.pipeline.transform.local_authority import make_str_slug
from asf_heat_pump_suitability.pipeline.validate.compare_versions import (
    COMPARISONS_DIR,
    STAGE_OUTPUT_DATASETS,
    get_str_stage_output_path,
)
from asf_heat_pump_suitability.utils import save_utils

STAGES = list(STAGE_OUTPUT_DATASETS)
SUMMARY_DIR = config["compare_versions"]["summary_dir"]
N_TOP_CHANGES = config["compare_versions"]["summary_n_top_changes"]
# Sort order of the statuses: compared rows first, then the rows with no change.
STATUSES = ["compared", "baseline", "comparison missing", "stage failed"]
# Up to this many LAs, the file name lists every LA slug.
MAX_LAS_NAMED_IN_SCOPE = 3

# JSON keys with a fixed set of sub-keys, flattened into one column each.
FLAT_KEYS = {
    "counts": [
        "rows_old",
        "rows_new",
        "rows_delta",
        "uprns_old",
        "uprns_new",
        "uprns_delta",
    ],
    "uprn_churn": [
        "n_added",
        "n_removed",
        "n_retained",
        "removed_share",
        "added_share",
        "n_null_old",
        "n_null_new",
    ],
    "cluster_counts": ["clusters_old", "clusters_new", "clusters_delta"],
    "total_area": ["area_m2_old", "area_m2_new", "area_m2_delta"],
}
MISSING_CLUSTERS_KEYS = [
    "uprns_in",
    "uprns_in_clusters",
    "uprns_missing",
    "missing_share",
]
# (metric name, old key, delta key) of the changes relative to the old value.
RELATIVE_CHANGE_METRICS = [
    ("rows", "rows_old", "rows_delta"),
    ("uprns", "uprns_old", "uprns_delta"),
    ("clusters", "clusters_old", "clusters_delta"),
    ("area_m2", "area_m2_old", "area_m2_delta"),
]
SHARE_METRICS = ["removed_share", "added_share"]
TECH_METRIC_PREFIX = "tech:"

FLOAT_COLUMNS = {
    "removed_share",
    "added_share",
    "area_m2_old",
    "area_m2_new",
    "area_m2_delta",
    "missing_share_old",
    "missing_share_new",
}
# The columns of the summary table. They are fixed, so a run with only
# baselines or missing rows still has every column.
SUMMARY_SCHEMA = {
    "local_authority": pl.String,
    "stage": pl.String,
    "status": pl.String,
    "max_rel_change": pl.Float64,
    "max_rel_change_metric": pl.String,
    "schema_version": pl.Int64,
    "release_date_old": pl.String,
    "release_date_new": pl.String,
    "trigger": pl.String,
    **{
        column: pl.Float64 if column in FLOAT_COLUMNS else pl.Int64
        for column in [
            *(sub_key for sub_keys in FLAT_KEYS.values() for sub_key in sub_keys),
            *(
                f"{key}_{version}"
                for version in ("old", "new")
                for key in MISSING_CLUSTERS_KEYS
            ),
        ]
    },
    "tech_counts": pl.List(
        pl.Struct(
            {
                "assigned_tech": pl.String,
                "n_old": pl.Int64,
                "n_new": pl.Int64,
                "n_delta": pl.Int64,
            }
        )
    ),
    "tolerance_warnings": pl.List(pl.String),
    "n_json_files": pl.Int64,
    "json_path": pl.String,
    "unused_json_paths": pl.List(pl.String),
}


def load_list_dict_comparison_files(
    local_authority: str, release_date: str
) -> list[dict]:
    """
    Load all comparison JSON files of one LA and release from S3.

    Args:
        local_authority (str): LA slug used in output paths
        release_date (str): release date, YYYYMMDD

    Returns:
        list[dict]: one dict for each JSON file, with keys:
            - path (str): S3 path of the file
            - last_modified (datetime): S3 write time of the file
            - summary (dict): the file's content
    """
    comparisons_dir = COMPARISONS_DIR.format(
        local_authority=local_authority, release_date=release_date
    )
    fs = s3fs.S3FileSystem()
    try:
        objects = fs.ls(comparisons_dir, detail=True)
    except FileNotFoundError:
        # An LA with no comparison for the release has no comparisons folder.
        return []
    files = []
    for obj in objects:
        if not obj["name"].endswith(".json"):
            continue
        with fs.open(obj["name"]) as f:
            summary = json.load(f)
        files.append(
            {
                "path": f"s3://{obj['name']}",
                "last_modified": obj["LastModified"],
                "summary": summary,
            }
        )
    return files


def get_bool_stage_output_exists(
    stage: str, local_authority: str, release_date: str
) -> bool:
    """
    Check if a stage wrote its output for one LA and release on S3.

    Args:
        stage (str): pipeline stage, a key of `STAGE_OUTPUT_DATASETS`
        local_authority (str): LA slug used in output paths
        release_date (str): release date, YYYYMMDD

    Returns:
        bool: True when the stage output exists
    """
    try:
        get_str_stage_output_path(
            stage=stage,
            local_authority=local_authority,
            release_date=release_date,
            check_exists=True,
        )
    except FileNotFoundError:
        return False
    return True


def generate_tuple_max_rel_change(summary: dict) -> tuple[float | None, str | None]:
    """
    Find the largest relative change in one comparison JSON.

    A metric whose old value is 0 or null is not used.

    Args:
        summary (dict): content of one comparison JSON

    Returns:
        tuple[float | None, str | None]: the largest absolute relative change
            and the name of its metric (e.g. "rows", "removed_share",
            "tech:Networked heat pump"), or (None, None) when no metric has
            a value
    """
    changes = []
    flat = {}
    for key in ("counts", "cluster_counts", "total_area"):
        flat.update(summary.get(key) or {})
    for metric, old_key, delta_key in RELATIVE_CHANGE_METRICS:
        if flat.get(old_key):
            changes.append((abs(flat[delta_key]) / flat[old_key], metric))
    churn = summary.get("uprn_churn") or {}
    for metric in SHARE_METRICS:
        if churn.get(metric) is not None:
            changes.append((abs(churn[metric]), metric))
    for tech in summary.get("tech_counts") or []:
        if tech["n_old"]:
            changes.append(
                (
                    abs(tech["n_delta"]) / tech["n_old"],
                    f"{TECH_METRIC_PREFIX}{tech['assigned_tech']}",
                )
            )
    # max() keeps the first metric of a tie, so the order above sets which
    # metric is named when changes are equal (e.g. all 0).
    return max(changes, key=lambda change: change[0], default=(None, None))


def _generate_dict_flat_summary(summary: dict) -> dict:
    """
    Flatten one comparison JSON into the columns of the summary table.

    A version 0 JSON has no schema_version, tech_counts or
    uprns_missing_clusters; those columns are null.

    Args:
        summary (dict): content of one comparison JSON

    Returns:
        dict: one value for each column of `SUMMARY_SCHEMA` that a JSON gives
    """
    max_rel_change, metric = generate_tuple_max_rel_change(summary=summary)
    row = {
        "status": summary["status"],
        "max_rel_change": max_rel_change,
        "max_rel_change_metric": metric,
        "schema_version": summary.get("schema_version", 0),
        "release_date_old": summary["release_date_old"],
        "release_date_new": summary["release_date_new"],
        "trigger": summary["trigger"],
        "tech_counts": summary.get("tech_counts"),
        "tolerance_warnings": summary["tolerance_warnings"],
    }
    for key, sub_keys in FLAT_KEYS.items():
        values = summary.get(key) or {}
        row.update({sub_key: values.get(sub_key) for sub_key in sub_keys})
    missing_clusters = summary.get("uprns_missing_clusters") or {}
    for version in ("old", "new"):
        values = missing_clusters.get(version) or {}
        row.update(
            {f"{key}_{version}": values.get(key) for key in MISSING_CLUSTERS_KEYS}
        )
    return row


def generate_list_dict_la_rows(
    local_authority: str, release_date: str, comparison_files: list[dict]
) -> list[dict]:
    """
    Make the summary rows of one LA, one for each stage.

    When a stage has more than one JSON, the one written last to S3 is used.
    A stage with no JSON is checked for its output on S3.

    Args:
        local_authority (str): LA slug used in output paths
        release_date (str): release date, YYYYMMDD
        comparison_files (list[dict]): the LA's JSON files, as from
            `load_list_dict_comparison_files`, with keys path, last_modified
            and summary

    Returns:
        list[dict]: one row for each stage, with the columns of
            `SUMMARY_SCHEMA`
    """
    rows = []
    for stage in STAGES:
        stage_files = sorted(
            (file for file in comparison_files if file["summary"]["stage"] == stage),
            key=lambda file: file["last_modified"],
            reverse=True,
        )
        row = {
            "local_authority": local_authority,
            "stage": stage,
            "n_json_files": len(stage_files),
            "unused_json_paths": [file["path"] for file in stage_files[1:]],
        }
        if stage_files:
            row["json_path"] = stage_files[0]["path"]
            row.update(_generate_dict_flat_summary(summary=stage_files[0]["summary"]))
        elif get_bool_stage_output_exists(
            stage=stage, local_authority=local_authority, release_date=release_date
        ):
            row["status"] = "comparison missing"
        else:
            row["status"] = "stage failed"
        rows.append(row)
    return rows


def generate_df_comparison_summary(rows: list[dict]) -> pl.DataFrame:
    """
    Make the summary table, sorted by the largest relative change.

    Compared rows come first, largest change first. Baselines and rows with
    no JSON come after them.

    Args:
        rows (list[dict]): summary rows, as from `generate_list_dict_la_rows`

    Returns:
        pl.DataFrame: one row for each LA and stage, with the columns of
            `SUMMARY_SCHEMA`
    """
    return (
        pl.DataFrame(rows, schema=SUMMARY_SCHEMA)
        .with_columns(
            status_order=pl.col("status").replace_strict(
                STATUSES, list(range(len(STATUSES)))
            ),
            stage_order=pl.col("stage").replace_strict(
                STAGES, list(range(len(STAGES)))
            ),
        )
        .sort(
            ["status_order", "max_rel_change", "local_authority", "stage_order"],
            descending=[False, True, False, False],
            nulls_last=True,
        )
        .drop("status_order", "stage_order")
    )


def get_str_summary_scope(la_slugs: list[str], is_gb: bool) -> str:
    """
    Name the LAs a summary covers, for its file names.

    Args:
        la_slugs (list[str]): slugs of the LAs in the summary
        is_gb (bool): True when the summary covers the full GB list

    Returns:
        str: "gb", the sorted slugs joined with "__" for up to
            `MAX_LAS_NAMED_IN_SCOPE` LAs, or "{n}_las_{first}_to_{last}"
    """
    if is_gb:
        return "gb"
    slugs = sorted(la_slugs)
    if len(slugs) <= MAX_LAS_NAMED_IN_SCOPE:
        return "__".join(slugs)
    return f"{len(slugs)}_las_{slugs[0]}_to_{slugs[-1]}"


def _generate_str_metric_cell(row: dict) -> str:
    """
    Name a row's largest change, with the old and new counts for a tech.

    Args:
        row (dict): one row of the summary table

    Returns:
        str: the metric name, e.g. "tech:Networked heat pump (10 -> 12)"
    """
    metric = row["max_rel_change_metric"]
    if not metric.startswith(TECH_METRIC_PREFIX):
        return metric
    tech_label = metric.removeprefix(TECH_METRIC_PREFIX)
    tech = next(t for t in row["tech_counts"] if t["assigned_tech"] == tech_label)
    return f"{metric} ({tech['n_old']:,} -> {tech['n_new']:,})"


def generate_str_summary_markdown(
    df_summary: pl.DataFrame,
    release_date: str,
    la_slugs: list[str],
    n_top_changes: int,
) -> str:
    """
    Write the summary table as a short markdown report for people.

    Args:
        df_summary (pl.DataFrame): summary table, as from
            `generate_df_comparison_summary`
        release_date (str): release date, YYYYMMDD
        la_slugs (list[str]): slugs of every LA in the run
        n_top_changes (int): number of compared rows to list as top changes

    Returns:
        str: the markdown report
    """
    lines = [
        f"# Comparison summary: release {release_date}",
        "",
        f"LAs in this summary ({len(la_slugs)}): {', '.join(sorted(la_slugs))}",
        "",
        "## Run health",
        "",
        f"| stage | {' | '.join(STATUSES)} |",
        f"|---|{'---|' * len(STATUSES)}",
    ]
    for stage, df_stage in [
        *((stage, df_summary.filter(pl.col("stage") == stage)) for stage in STAGES),
        ("all", df_summary),
    ]:
        counts = [str((df_stage["status"] == status).sum()) for status in STATUSES]
        lines.append(f"| {stage} | {' | '.join(counts)} |")

    df_missing = df_summary.filter(
        pl.col("status").is_in(["comparison missing", "stage failed"])
    )
    lines += ["", "## Rows with no comparison", ""]
    if df_missing.is_empty():
        lines.append("None.")
    else:
        lines += ["| LA | stage | status |", "|---|---|---|"]
        lines += [
            f"| {row['local_authority']} | {row['stage']} | {row['status']} |"
            for row in df_missing.iter_rows(named=True)
        ]

    df_top = df_summary.filter(pl.col("status") == "compared").head(n_top_changes)
    lines += [
        "",
        "## Top changes",
        "",
        f"The {len(df_top)} compared rows with the largest relative change "
        "since the older release.",
        "",
        "| LA | stage | releases | change | metric | report |",
        "|---|---|---|---|---|---|",
    ]
    for row in df_top.iter_rows(named=True):
        change = (
            "-" if row["max_rel_change"] is None else f"{row['max_rel_change']:.2%}"
        )
        metric = (
            "-"
            if row["max_rel_change_metric"] is None
            else _generate_str_metric_cell(row=row)
        )
        report_path = row["json_path"].removesuffix(".json") + ".md"
        lines.append(
            f"| {row['local_authority']} | {row['stage']} | "
            f"{row['release_date_old']} -> {row['release_date_new']} | "
            f"{change} | {metric} | {report_path} |"
        )

    warnings = [
        f"- {row['local_authority']} {row['stage']}: {warning}"
        for row in df_summary.iter_rows(named=True)
        for warning in row["tolerance_warnings"] or []
    ]
    lines += ["", "## Tolerance warnings", "", *(warnings or ["None."])]

    unused_paths = [
        f"- {path}" for paths in df_summary["unused_json_paths"] for path in paths
    ]
    lines += [
        "",
        "## JSON files not used",
        "",
        "An LA and stage with more than one JSON uses the one written last.",
        "",
        *(unused_paths or ["None."]),
    ]
    return "\n".join(lines) + "\n"


def upload_summary_files_to_s3(file_paths: list[Path], release_date: str) -> None:
    """
    Upload the summary files to the release's folder in `SUMMARY_DIR`.

    Args:
        file_paths (list[Path]): local parquet and markdown files
        release_date (str): release date, YYYYMMDD
    """
    bucket, key_dir = (
        SUMMARY_DIR.format(release_date=release_date)
        .removeprefix("s3://")
        .split("/", 1)
    )
    for path in file_paths:
        # upload_file_to_s3 builds the key as f"{s3_key_dir}{subfolder}/{filename}",
        # so an empty subfolder puts the file straight into key_dir.
        save_utils.upload_file_to_s3(
            local_file_path=str(path),
            s3_bucket=bucket,
            s3_key_dir=key_dir,
            filename=path.name,
            subfolder="",
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Summarise the comparison JSON files of one release."
    )
    parser.add_argument(
        "--release_date",
        required=True,
        help="Release date of the comparisons, YYYYMMDD.",
    )
    parser.add_argument(
        "--local_authorities",
        nargs="+",
        required=True,
        help="LA names (case insensitive), or GB for every LA in GB.",
    )
    parser.add_argument(
        "--report_dir",
        default=str(PROJECT_DIR / "outputs" / "comparisons"),
        help="Directory the parquet and markdown files are written to.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also upload both files to the release's summary folder on S3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    release_date = save_utils.get_str_release_date(args.release_date)
    # Same GB rule as orbit_pipeline.py, so a summary expects the LAs a GB run starts.
    is_gb = args.local_authorities == ["GB"]
    la_names = (
        pl.read_csv(config["data"]["processed"]["valid_la_names"])["LAD23NM"].to_list()
        if is_gb
        else args.local_authorities
    )
    la_slugs = sorted({make_str_slug(name) for name in la_names})

    rows = []
    for local_authority in la_slugs:
        logging.info("Reading comparisons for %s", local_authority)
        rows += generate_list_dict_la_rows(
            local_authority=local_authority,
            release_date=release_date,
            comparison_files=load_list_dict_comparison_files(
                local_authority=local_authority, release_date=release_date
            ),
        )
    df_summary = generate_df_comparison_summary(rows=rows)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"comparison_summary_{get_str_summary_scope(la_slugs=la_slugs, is_gb=is_gb)}"
    )
    parquet_path = report_dir / f"{file_stem}.parquet"
    markdown_path = report_dir / f"{file_stem}.md"
    df_summary.write_parquet(parquet_path)
    markdown_path.write_text(
        generate_str_summary_markdown(
            df_summary=df_summary,
            release_date=release_date,
            la_slugs=la_slugs,
            n_top_changes=N_TOP_CHANGES,
        )
    )
    logging.info("Summary written to %s and %s", parquet_path, markdown_path)
    if args.save:
        upload_summary_files_to_s3(
            file_paths=[parquet_path, markdown_path], release_date=release_date
        )
    for status, n_rows in (
        df_summary.group_by("status").len().sort("status").iter_rows()
    ):
        logging.info("%s: %d rows", status, n_rows)
