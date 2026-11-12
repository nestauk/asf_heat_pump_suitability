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

    Args:
        mocker (pytest_mock.MockerFixture): mocks the S3 stage-output check

    Returns:
        pl.DataFrame: the summary table, 10 rows
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

    tech_counts = _row(df_summary, "east_lothian", "decision_tree")["tech_counts"]
    assert [tech["n_delta"] for tech in tech_counts] == [
        0,
        0,
        2,
        -2,
        0,
    ], "tech_counts should be kept as one nested cell with each tech's counts"


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


def test_summary_lists_compared_rows_first_in_pipeline_order(df_summary):
    """Compared rows come first, then the rest; each LA's stages in pipeline order."""
    # The rows in reverse order (Midlothian first, stages backwards), so the
    # test passes only if the table is really sorted.
    df = summarise_comparisons.generate_df_comparison_summary(
        rows=df_summary.reverse().to_dicts()
    )
    assert (
        df["status"].to_list()[:5] == ["compared"] * 5
    ), "All compared rows should come before the rows with no JSON"
    assert df["stage"].to_list()[:5] == list(
        compare_versions.STAGE_OUTPUT_DATASETS
    ), "An LA's stages should be in pipeline order"


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


def test_summary_markdown_gives_run_health(df_summary):
    """The markdown shows the status counts for each stage and in all."""
    markdown = summarise_comparisons.generate_str_summary_markdown(
        df_summary=df_summary,
        release_date=RELEASE_DATE,
        la_slugs=["east_lothian", "midlothian"],
    )
    expected_lines = [
        # Run-health table: stage, compared, comparison missing, stage failed
        "| uprns | 1 | 0 | 1 |",
        "| all | 5 | 1 | 4 |",
    ]
    for line in expected_lines:
        assert line in markdown, f"The markdown should contain the line {line!r}"


def test_reader_keys_match_what_compare_versions_writes():
    """The keys this script reads are exactly the keys compare_versions writes."""
    # One small output with every column the writer reads: UPRNs with a tech,
    # in two clusters of the clusters layer.
    df = pl.DataFrame(
        {
            "UPRN": [1, 2, 3],
            "assigned_tech": ["A", "B", "A"],
            "cluster_id": ["HP_1", "HP_1", "HP_2"],
            "n_UPRNs": [2.0, 1.0, 0.0],
            "layer": [compare_versions.CLUSTER_LAYER] * 3,
        }
    )
    df_areas = pl.DataFrame(
        {"area_m2": [10.0, 20.0, 30.0], "layer": [compare_versions.CLUSTER_LAYER] * 3}
    )
    # Between them, these two stages write every key of the JSON.
    written = {}
    for stage in ("decision_tree", "compute_contextual_features"):
        written.update(
            compare_versions.generate_dict_comparison_summary(
                stage=stage,
                local_authority="east_lothian",
                trigger=None,
                release_date_old="20261013",
                release_date_new=RELEASE_DATE,
                df_old=df,
                df_new=df.head(2),
                df_areas_old=df_areas,
                df_areas_new=df_areas,
                df_uprns_old=df.select("UPRN"),
                df_uprns_new=df.select("UPRN"),
            )
        )
    for key, sub_keys in summarise_comparisons.FLAT_KEYS.items():
        assert set(written[key]) == set(sub_keys), (
            f"compare_versions writes {sorted(written[key])} under {key!r}, "
            f"but the reader expects {sorted(sub_keys)}"
        )
    assert set(written["uprns_missing_clusters"]["new"]) == set(
        summarise_comparisons.MISSING_CLUSTERS_KEYS
    ), "The missing-cluster keys written and read should be the same"
    tech_fields = {
        field.name
        for field in summarise_comparisons.SUMMARY_SCHEMA["tech_counts"].inner.fields
    }
    assert (
        set(written["tech_counts"][0]) == tech_fields
    ), "The tech_counts fields written and read should be the same"


def test_summary_stops_on_a_json_of_another_schema_version(mocker):
    """A JSON with a layout the reader does not know stops the run."""
    mocker.patch.object(
        summarise_comparisons, "get_bool_stage_output_exists", return_value=False
    )
    summary = _load_fixture("uprns_east_lothian_20261013_vs_20261014.json")
    summary["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version 3"):
        summarise_comparisons.generate_list_dict_la_rows(
            local_authority="east_lothian",
            release_date=RELEASE_DATE,
            comparison_files=[
                _comparison_file(path=f"{EL_DIR}/uprns.json", summary=summary, minute=1)
            ],
        )
