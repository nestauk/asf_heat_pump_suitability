"""
Tests for asf_heat_pump_suitability.pipeline.validate.summarise_comparisons.

The JSON fixtures are the real East Lothian comparison files of release
20261014 (all five stages), in the version 2 layout: no status key, and
only the keys that apply to each stage.

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
EL_DIR = "s3://bucket/outputs/data/east_lothian/20261014/compare_versions"


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

    East Lothian has the five real JSONs, plus an older second add_features
    JSON. Midlothian has no JSON; of its stage outputs, only the cluster
    output exists.
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
    mocker.patch.object(
        summarise_comparisons,
        "get_bool_stage_output_exists",
        side_effect=lambda stage, local_authority, release_date: stage == "cluster",
    )
    rows = []
    for local_authority, files in [
        ("east_lothian", el_files),
        ("midlothian", []),
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
    ), "Every East Lothian stage has a JSON, so every stage is compared"
    assert (
        statuses[("midlothian", "cluster")] == "comparison missing"
    ), "No JSON but a stage output should show as comparison missing"
    assert (
        statuses[("midlothian", "add_features")] == "stage failed"
    ), "No JSON and no stage output should show as stage failed"


def test_summary_flattens_the_keys_each_stage_has(df_summary):
    """Each JSON key becomes columns; a key the stage does not have is null."""
    row = _row(df_summary, "east_lothian", "compute_contextual_features")
    assert (
        row["schema_version"],
        row["clusters_new"],
        row["uprns_missing_new"],
        row["n_added"],
    ) == (2, 5346, 292, None), (
        "Fixed-shape keys should be flattened into columns, and uprn_churn, "
        "which this stage has not, should give null columns"
    )

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
        df_summary["status"].to_list()[:5] == ["compared"] * 5
    ), "All compared rows should come before the rows with no JSON"
    changes = df_summary["max_rel_change"].to_list()[:5]
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
    "stage, key, edit, expected_metric",
    [
        # 50 new UPRNs with no tech, none before: no relative change, skipped.
        (
            "decision_tree",
            "tech_counts",
            {"n_old": 0, "n_new": 50, "n_delta": 50},
            "tech:Individual or Networked heat pump",
        ),
        # Same total, but half the old UPRNs replaced: churn beats the techs.
        ("decision_tree", "uprn_churn", {"removed_share": 0.5}, "removed_share"),
        # Clusters and area flat, but 12% of UPRNs now have no cluster (0.5%
        # before): the change in missing_share wins.
        (
            "compute_contextual_features",
            "uprns_missing_clusters",
            {"missing_share": 0.12},
            "missing_share",
        ),
    ],
)
def test_max_rel_change_skips_zero_old_values_and_uses_shares(
    stage, key, edit, expected_metric
):
    """A metric with old = 0 is not used; changes in shares count as changes."""
    summary = _load_fixture(f"{stage}_east_lothian_20261013_vs_20261014.json")
    # The first tech_counts entry is the "(null)" tech. The missing-cluster
    # edit is made to the new version.
    if key == "tech_counts":
        target = summary[key][0]
    elif key == "uprns_missing_clusters":
        target = summary[key]["new"]
    else:
        target = summary[key]
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
    """The markdown shows the status counts and the top changes."""
    markdown = summarise_comparisons.generate_str_summary_markdown(
        df_summary=df_summary,
        release_date=RELEASE_DATE,
        la_slugs=["east_lothian", "midlothian"],
        n_top_changes=3,
        min_rel_change=0.5,
    )
    expected_lines = [
        # Run-health table: stage, compared, comparison missing, stage failed
        "| uprns | 1 | 0 | 1 |",
        "| all | 5 | 1 | 4 |",
    ]
    for line in expected_lines:
        assert line in markdown, f"The markdown should contain the line {line!r}"
    assert (
        "| 0.0203% | tech:Individual or Networked heat pump (9,831 -> 9,833)"
        in markdown
        and f"{EL_DIR}/decision_tree_east_lothian_20261013_vs_20261014.md" in markdown
    ), "The top change should show its size to 3 significant figures, the tech counts and the per-LA report path"


@pytest.mark.parametrize(
    "n_top_changes, min_rel_change, expected_rows",
    [
        # East Lothian's changes: 0.0203%, 0.0186%, 0.0092%, 0, 0.
        # Two rows reach 0.01%, so both are listed although the floor is 1.
        (1, 0.0001, 2),
        # No row reaches 50%, so the floor of 3 rows still applies.
        (3, 0.5, 3),
    ],
)
def test_top_changes_list_every_row_over_the_limit_and_at_least_n(
    df_summary, n_top_changes, min_rel_change, expected_rows
):
    """Top changes have no cap above the limit, and a floor of rows below it."""
    markdown = summarise_comparisons.generate_str_summary_markdown(
        df_summary=df_summary,
        release_date=RELEASE_DATE,
        la_slugs=["east_lothian", "midlothian"],
        n_top_changes=n_top_changes,
        min_rel_change=min_rel_change,
    )
    assert markdown.count("| east_lothian | ") == expected_rows, (
        f"Expected {expected_rows} top-change rows: every row at or over "
        f"{min_rel_change:.2%}, and at least {n_top_changes}"
    )
