"""
Tests for asf_heat_pump_suitability.pipeline.validate.summarise_comparisons.

The JSON fixtures are copies of the real East Lothian comparison files:
version 1 (release 20261014, all five stages) and version 0 (release
20261013, decision_tree only).

Run:
pytest asf_heat_pump_suitability/pipeline/validate/tests/test_summarise_comparisons.py
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from asf_heat_pump_suitability.pipeline.validate import (
    compare_versions,
    summarise_comparisons,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "comparisons"
RELEASE_DATE = "20261014"
EL_DIR = "s3://bucket/outputs/data/east_lothian/20261014/comparisons"
ML_DIR = "s3://bucket/outputs/data/midlothian/20261014/comparisons"


def _load_fixture(name: str) -> dict:
    """
    Load one comparison JSON fixture.

    Args:
        name (str): file name in `FIXTURE_DIR`

    Returns:
        dict: the comparison summary
    """
    return json.loads((FIXTURE_DIR / name).read_text())


def _comparison_file(path: str, summary: dict, minute: int) -> dict:
    """
    Wrap a summary as `load_list_dict_comparison_files` returns it.

    Args:
        path (str): S3 path of the JSON file
        summary (dict): the comparison summary
        minute (int): minute of the S3 write time, to order duplicates

    Returns:
        dict: keys path, last_modified and summary
    """
    return {
        "path": path,
        "last_modified": datetime(2026, 10, 14, 9, minute, tzinfo=timezone.utc),
        "summary": summary,
    }


@pytest.fixture
def df_summary(mocker):
    """
    Summary table for two LAs.

    East Lothian has the five real version 1 JSONs, plus an older second
    add_features JSON. Midlothian has the real version 0 decision_tree JSON,
    a baseline uprns JSON and no JSON for the other stages; of those, only
    the cluster output exists.
    """
    el_files = [
        _comparison_file(
            path=f"{EL_DIR}/{path.name}",
            summary=json.loads(path.read_text()),
            minute=27,
        )
        for path in sorted(FIXTURE_DIR.glob("*_20261013_vs_20261014.json"))
    ]
    add_features_old = copy.deepcopy(
        _load_fixture("add_features_east_lothian_20261013_vs_20261014.json")
    )
    add_features_old["release_date_old"] = "20260907"
    el_files.append(
        _comparison_file(
            path=f"{EL_DIR}/add_features_east_lothian_20260907_vs_20261014.json",
            summary=add_features_old,
            minute=5,
        )
    )
    decision_tree_v0 = _load_fixture(
        "decision_tree_east_lothian_20260907_vs_20261013.json"
    )
    decision_tree_v0["local_authority"] = "midlothian"
    ml_files = [
        _comparison_file(
            path=f"{ML_DIR}/decision_tree_v0.json", summary=decision_tree_v0, minute=1
        ),
        _comparison_file(
            path=f"{ML_DIR}/uprns_baseline.json",
            summary=compare_versions.generate_dict_comparison_summary(
                stage="uprns",
                local_authority="midlothian",
                trigger=None,
                release_date_new=RELEASE_DATE,
            ),
            minute=1,
        ),
    ]
    mocker.patch.object(
        summarise_comparisons,
        "get_bool_stage_output_exists",
        side_effect=lambda stage, local_authority, release_date: stage == "cluster",
    )
    rows = []
    for local_authority, files in [
        ("east_lothian", el_files),
        ("midlothian", ml_files),
    ]:
        rows += summarise_comparisons.generate_list_dict_la_rows(
            local_authority=local_authority,
            release_date=RELEASE_DATE,
            comparison_files=files,
        )
    return summarise_comparisons.generate_df_comparison_summary(rows=rows)


def _row(df: pl.DataFrame, local_authority: str, stage: str) -> dict:
    """
    Get one LA and stage row of the summary table.

    Args:
        df (pl.DataFrame): summary table
        local_authority (str): LA slug
        stage (str): pipeline stage

    Returns:
        dict: the row
    """
    return df.filter(
        (pl.col("local_authority") == local_authority) & (pl.col("stage") == stage)
    ).row(0, named=True)


def test_summary_has_one_row_per_expected_la_and_stage_with_its_status(df_summary):
    """Each LA gets every stage, and a missing JSON is labelled by its output."""
    statuses = dict(
        zip(
            zip(df_summary["local_authority"], df_summary["stage"]),
            df_summary["status"],
        )
    )
    stages = list(compare_versions.STAGE_OUTPUT_DATASETS)
    assert len(df_summary) == 2 * len(stages) and set(statuses) == {
        (la, stage) for la in ("east_lothian", "midlothian") for stage in stages
    }, "There should be exactly one row for each LA and stage"
    assert all(
        statuses[("east_lothian", stage)] == "compared" for stage in stages
    ), "Every East Lothian stage has a JSON with status compared"
    assert (
        statuses[("midlothian", "uprns")] == "baseline"
    ), "A baseline JSON should show as baseline, not as a change"
    assert (
        statuses[("midlothian", "cluster")] == "comparison missing"
    ), "No JSON but a stage output should show as comparison missing"
    assert (
        statuses[("midlothian", "add_features")] == "stage failed"
    ), "No JSON and no stage output should show as stage failed"


def test_summary_flattens_version_0_and_version_1_json(df_summary):
    """Both JSON versions are read; keys version 0 does not have are null."""
    v0 = _row(df_summary, "midlothian", "decision_tree")
    assert (v0["schema_version"], v0["rows_old"], v0["tech_counts"]) == (
        0,
        58674,
        None,
    ), "A JSON with no schema_version is version 0 and has no tech counts"

    v1 = _row(df_summary, "east_lothian", "compute_contextual_features")
    assert (
        v1["clusters_new"],
        v1["uprns_missing_new"],
        v1["n_added"],
    ) == (5346, 292, None), "Fixed-shape keys should be flattened into columns"

    techs = (
        df_summary.filter(pl.col("stage") == "decision_tree")
        .filter(pl.col("local_authority") == "east_lothian")
        .explode("tech_counts")
        .unnest("tech_counts")
    )
    assert techs.filter(pl.col("assigned_tech") == "Individual solution")[
        "n_delta"
    ].to_list() == [-2], "tech_counts should explode to one row for each tech"


def test_summary_uses_the_newest_of_two_json_files(df_summary):
    """A second JSON is counted and listed, and the newest one is used."""
    row = _row(df_summary, "east_lothian", "add_features")
    assert (
        row["n_json_files"],
        row["release_date_old"],
        row["unused_json_paths"],
    ) == (
        2,
        "20261013",
        [f"{EL_DIR}/add_features_east_lothian_20260907_vs_20261014.json"],
    ), "The JSON written last should be used and the other one listed"


def test_summary_is_sorted_by_largest_relative_change(df_summary):
    """Compared rows come first, largest change first; the rest come after."""
    assert (
        df_summary["status"].to_list()[:6] == ["compared"] * 6
    ), "All compared rows should come before baselines and missing rows"
    changes = df_summary["max_rel_change"].to_list()[:6]
    assert changes == sorted(
        changes, reverse=True
    ), "Compared rows should be sorted by max_rel_change, largest first"
    top = df_summary.row(0, named=True)
    assert (top["stage"], top["max_rel_change_metric"]) == (
        "decision_tree",
        "tech:Individual or Networked heat pump",
    ), "East Lothian's top change is 2 of 9,831 UPRNs moving tech"
    assert top["max_rel_change"] == pytest.approx(
        2 / 9831
    ), "max_rel_change should be |n_delta| / n_old of that tech"


@pytest.mark.parametrize(
    "key, edit, expected_metric",
    [
        # 50 new UPRNs with no tech, none before: no relative change, skipped.
        (
            "tech_counts",
            {"n_old": 0, "n_new": 50, "n_delta": 50},
            "tech:Individual or Networked heat pump",
        ),
        # Same total, but half the old UPRNs replaced: churn beats the techs.
        ("uprn_churn", {"removed_share": 0.5}, "removed_share"),
    ],
)
def test_max_rel_change_skips_zero_old_values_and_uses_churn(
    key, edit, expected_metric
):
    """A metric with old = 0 is not used; churn shares count as changes."""
    summary = _load_fixture("decision_tree_east_lothian_20261013_vs_20261014.json")
    # The first tech_counts entry is the "(null)" tech.
    target = summary[key][0] if key == "tech_counts" else summary[key]
    target.update(edit)
    _, metric = summarise_comparisons.generate_tuple_max_rel_change(summary=summary)
    assert metric == expected_metric, f"The largest change should be {expected_metric}"


@pytest.mark.parametrize(
    "la_slugs, is_gb, expected",
    [
        (["plymouth", "east_lothian"], True, "gb"),
        (["glasgow_city", "east_lothian"], False, "east_lothian__glasgow_city"),
        (
            [f"la_{i:02d}" for i in range(12, 0, -1)],
            False,
            "12_las_la_01_to_la_12",
        ),
    ],
)
def test_summary_scope_names_the_las_covered(la_slugs, is_gb, expected):
    """File names say which LAs a summary covers, in any input order."""
    assert (
        summarise_comparisons.get_str_summary_scope(la_slugs=la_slugs, is_gb=is_gb)
        == expected
    ), f"The scope for {la_slugs} should be {expected}"


def test_summary_markdown_gives_run_health_and_top_changes(df_summary):
    """The markdown shows status counts, missing rows, top changes and extras."""
    markdown = summarise_comparisons.generate_str_summary_markdown(
        df_summary=df_summary,
        release_date=RELEASE_DATE,
        la_slugs=["east_lothian", "midlothian"],
        n_top_changes=3,
    )
    expected_lines = [
        # Run-health table: stage, compared, baseline, comparison missing, stage failed
        "| uprns | 1 | 1 | 0 | 0 |",
        "| all | 6 | 1 | 1 | 2 |",
        "| midlothian | add_features | stage failed |",
        f"- {EL_DIR}/add_features_east_lothian_20260907_vs_20261014.json",
    ]
    for line in expected_lines:
        assert line in markdown, f"The markdown should contain the line {line!r}"
    assert (
        "tech:Individual or Networked heat pump (9,831 -> 9,833)" in markdown
        and f"{EL_DIR}/decision_tree_east_lothian_20261013_vs_20261014.md" in markdown
    ), "The top change should show the tech counts and the per-LA report path"
    assert (
        markdown.count("| east_lothian | ") == 3
    ), "Only n_top_changes rows should be listed as top changes"


def test_upload_summary_files_to_s3_writes_to_release_folder(mocker):
    """--save puts both files in outputs/comparisons/{release_date}/."""
    upload = mocker.patch(
        "asf_heat_pump_suitability.utils.save_utils.upload_file_to_s3"
    )
    summarise_comparisons.upload_summary_files_to_s3(
        file_paths=[Path("/tmp/comparison_summary_gb.parquet")],
        release_date=RELEASE_DATE,
    )
    kwargs = upload.call_args.kwargs
    expected_dir = summarise_comparisons.SUMMARY_DIR.format(release_date=RELEASE_DATE)
    assert (
        f"s3://{kwargs['s3_bucket']}/{kwargs['s3_key_dir']}{kwargs['subfolder']}/"
        f"{kwargs['filename']}" == f"{expected_dir}/comparison_summary_gb.parquet"
    ), "The summary should be uploaded to the release's comparisons folder"
