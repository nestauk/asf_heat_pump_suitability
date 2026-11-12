"""
Summarise the comparison JSON files of many local authorities for one release.

`pipeline.py --compare` writes one comparison JSON for each LA and stage to
`outputs/data/{la}/{release_date}/compare_versions/`. This script reads them
for the given LAs and gives one row for each LA and stage, with a status:

- compared: a JSON compares the release with an older one
- comparison missing: no JSON, but the stage output for the release exists.
  The comparison failed, did not run, or had no older release to compare with
- stage failed: no JSON and no stage output for the release

Rows are sorted by status, then LA, then stage. The script writes a parquet
table and a markdown summary named for the LAs they cover, `comparison_summary_{scope}.parquet` and `.md`. Pass
--save to also upload both to `outputs/compare_versions/{release_date}/` on S3.

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
from asf_heat_pump_suitability.pipeline.transform.local_authority import (
    load_list_la_names,
    make_str_slug,
    resolve_list_la_names,
)
from asf_heat_pump_suitability.pipeline.validate.compare_versions import (
    COMPARISONS_DIR,
    SUMMARY_SCHEMA_VERSION,
    STAGE_OUTPUT_DATASETS,
    get_str_stage_output_path,
)
from asf_heat_pump_suitability.utils import save_utils

STAGES = list(STAGE_OUTPUT_DATASETS)
SUMMARY_DIR = config["output"]["compare_versions_summary"]
# Sort order of the statuses: compared rows first, then the rows with no JSON.
STATUSES = ["compared", "comparison missing", "stage failed"]
# Up to this many LAs, the output file name lists every LA slug.
MAX_LAS_IN_FILE_NAME = 3

# JSON keys with a fixed set of sub-keys, flattened into one column each,
# with the column type.
FLAT_KEYS = {
    "counts": {
        "rows_old": pl.Int64,
        "rows_new": pl.Int64,
        "rows_delta": pl.Int64,
        "uprns_old": pl.Int64,
        "uprns_new": pl.Int64,
        "uprns_delta": pl.Int64,
    },
    "uprn_churn": {
        "n_added": pl.Int64,
        "n_removed": pl.Int64,
        "n_retained": pl.Int64,
        "removed_share": pl.Float64,
        "added_share": pl.Float64,
        "n_null_old": pl.Int64,
        "n_null_new": pl.Int64,
    },
    "cluster_counts": {
        "clusters_old": pl.Int64,
        "clusters_new": pl.Int64,
        "clusters_delta": pl.Int64,
    },
    "total_area": {
        "area_m2_old": pl.Float64,
        "area_m2_new": pl.Float64,
        "area_m2_delta": pl.Float64,
    },
}
# Sub-keys of each version ("old", "new") in uprns_missing_clusters, with the
# column type.
MISSING_CLUSTERS_KEYS = {
    "uprns_in": pl.Int64,
    "uprns_in_clusters": pl.Int64,
    "uprns_missing": pl.Int64,
    "missing_share": pl.Float64,
}
# The columns of the summary table. They are fixed, so a run with only
# missing rows still has every column.
SUMMARY_SCHEMA = {
    "local_authority": pl.String,
    "stage": pl.String,
    "status": pl.String,
    "schema_version": pl.Int64,
    "release_date_old": pl.String,
    "release_date_new": pl.String,
    "trigger": pl.String,
    **{
        sub_key: dtype
        for sub_keys in FLAT_KEYS.values()
        for sub_key, dtype in sub_keys.items()
    },
    **{
        f"{key}_{version}": dtype
        for version in ("old", "new")
        for key, dtype in MISSING_CLUSTERS_KEYS.items()
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


def generate_dict_flat_summary(summary: dict) -> dict:
    """
    Flatten one comparison JSON into the columns of the summary table.

    A JSON has only the keys that apply to its stage. The columns of the
    other keys are null.

    Args:
        summary (dict): content of one comparison JSON

    Returns:
        dict: one value for each column of `SUMMARY_SCHEMA` that a JSON
            gives
    """
    row = {
        "schema_version": summary["schema_version"],
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

    Raises:
        ValueError: if a JSON used has a schema_version other than
            `SUMMARY_SCHEMA_VERSION`
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
            newest = stage_files[0]
            # The version comes from compare_versions, so a new JSON layout
            # stops the run here and does not give empty columns.
            if newest["summary"]["schema_version"] != SUMMARY_SCHEMA_VERSION:
                raise ValueError(
                    f"{newest['path']} has schema_version "
                    f"{newest['summary']['schema_version']}, but this script reads "
                    f"version {SUMMARY_SCHEMA_VERSION} only."
                )
            # Every JSON is a real comparison: compare_versions writes none
            # when there is no older release.
            row["status"] = "compared"
            row["json_path"] = newest["path"]
            row.update(generate_dict_flat_summary(summary=newest["summary"]))
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
    Make the summary table, sorted by status, then LA, then stage.

    Compared rows come first. Rows with no JSON come after them.

    Args:
        rows (list[dict]): summary rows, as from `generate_list_dict_la_rows`

    Returns:
        pl.DataFrame: one row for each LA and stage, with the columns of
            `SUMMARY_SCHEMA`
    """
    # An Enum sorts in the order of its categories, so statuses and stages
    # sort in list order, not alphabetically.
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA).sort(
        pl.col("status").cast(pl.Enum(STATUSES)),
        "local_authority",
        pl.col("stage").cast(pl.Enum(STAGES)),
    )


def get_str_summary_scope(la_slugs: list[str], is_gb: bool) -> str:
    """
    Name the LAs a summary covers, for its file names.

    Args:
        la_slugs (list[str]): slugs of the LAs in the summary
        is_gb (bool): True when the summary covers the full GB list

    Returns:
        str: "gb", the sorted slugs joined with "__" for up to
            `MAX_LAS_IN_FILE_NAME` LAs, or "{n}_las_{first}_to_{last}"
    """
    if is_gb:
        return "gb"
    slugs = sorted(la_slugs)
    if len(slugs) <= MAX_LAS_IN_FILE_NAME:
        return "__".join(slugs)
    return f"{len(slugs)}_las_{slugs[0]}_to_{slugs[-1]}"


def generate_str_summary_markdown(
    df_summary: pl.DataFrame,
    release_date: str,
    la_slugs: list[str],
) -> str:
    """
    Write the summary table as a short markdown report for people.

    Args:
        df_summary (pl.DataFrame): summary table, as from
            `generate_df_comparison_summary`
        release_date (str): release date, YYYYMMDD
        la_slugs (list[str]): slugs of every LA in the run

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

    df_missing = df_summary.filter(pl.col("status") != "compared")
    lines += ["", "## Rows with no comparison", ""]
    if df_missing.is_empty():
        lines.append("None.")
    else:
        lines += ["| LA | stage | status |", "|---|---|---|"]
        lines += [
            f"| {row['local_authority']} | {row['stage']} | {row['status']} |"
            for row in df_missing.iter_rows(named=True)
        ]

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
    # Stops on a name that is not an LA, so a typo is not reported as
    # "stage failed". Gives None for GB, in any case.
    resolved_la_names = resolve_list_la_names(la_names=args.local_authorities)
    is_gb = resolved_la_names is None
    # The same GB list as orbit_pipeline.py, so a summary expects the LAs a GB
    # run starts.
    la_names = load_list_la_names(nation="GB") if is_gb else resolved_la_names
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
