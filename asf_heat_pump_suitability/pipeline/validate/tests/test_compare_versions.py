"""
Tests for asf_heat_pump_suitability.pipeline.validate.compare_versions.

Run:
pytest asf_heat_pump_suitability/pipeline/validate/tests/test_compare_versions.py
"""

import json
import subprocess

import polars as pl
import pytest

from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.pipeline.validate import compare_versions
from asf_heat_pump_suitability.utils import manifest_utils


@pytest.fixture(scope="module")
def df_old():
    """Old-version stage output: four UPRNs with a tech assignment."""
    return pl.DataFrame(
        {
            "UPRN": [1, 2, 3, 4],
            "assigned_tech": [
                "Individual solution",
                "Individual solution",
                "District heat network",
                "Communal solution",
            ],
            "in_hn_zone": [False, False, True, False],
        }
    )


@pytest.fixture(scope="module")
def df_new_identical(df_old):
    """New-version output identical to the old one (no drift)."""
    return df_old.clone()


@pytest.fixture(scope="module")
def df_new_churned():
    """New-version output with churn: UPRN 4 dropped, 5 and 6 added, and
    UPRN 3 moved from District heat network to Individual solution."""
    return pl.DataFrame(
        {
            "UPRN": [1, 2, 3, 5, 6],
            "assigned_tech": [
                "Individual solution",
                "Individual solution",
                "Individual solution",
                "Networked heat pump",
                "Networked heat pump",
            ],
            "in_hn_zone": [False, False, True, False, True],
        }
    )


class TestGenerateDictCountDelta:
    """Tests for `generate_dict_count_delta`."""

    def test_identical_versions_have_zero_deltas(self, df_old, df_new_identical):
        """No drift means zero row and UPRN deltas."""
        counts = compare_versions.generate_dict_count_delta(df_old, df_new_identical)
        assert counts["rows_delta"] == 0, "identical versions must show no row change"
        assert counts["uprns_delta"] == 0, "identical versions must show no UPRN change"

    def test_reports_old_new_and_delta_counts(self, df_old, df_new_churned):
        """Row and distinct-UPRN counts are reported for both versions."""
        counts = compare_versions.generate_dict_count_delta(df_old, df_new_churned)
        assert counts == {
            "rows_old": 4,
            "rows_new": 5,
            "rows_delta": 1,
            "uprns_old": 4,
            "uprns_new": 5,
            "uprns_delta": 1,
        }, "old, new and delta must be reported for both rows and distinct UPRNs"

    def test_uprn_counts_are_none_without_uprn_column(self, df_old):
        """Outputs without a UPRN column (e.g. clusters) get row counts only."""
        no_uprn = df_old.drop("UPRN")
        counts = compare_versions.generate_dict_count_delta(no_uprn, no_uprn)
        assert (
            counts["rows_delta"] == 0
        ), "row counts must still work without a UPRN column"
        assert counts["uprns_old"] is None, "no UPRN column means no old UPRN count"
        assert counts["uprns_new"] is None, "no UPRN column means no new UPRN count"
        assert counts["uprns_delta"] is None, "no UPRN column means no UPRN delta"


class TestGenerateDictSchemaDiff:
    """Tests for `generate_dict_schema_diff`."""

    def test_identical_schemas_have_empty_diff(self, df_old, df_new_identical):
        """No drift means no added, removed or retyped columns."""
        assert compare_versions.generate_dict_schema_diff(df_old, df_new_identical) == {
            "added": {},
            "removed": {},
            "dtype_changed": {},
        }, "identical schemas must yield an empty diff in every category"

    def test_reports_added_and_removed_columns_with_dtypes(self, df_old):
        """Column additions and removals are reported with their dtypes."""
        df_new = df_old.drop("in_hn_zone").with_columns(
            pl.lit(1.5).alias("garden_area_m2")
        )
        diff = compare_versions.generate_dict_schema_diff(df_old, df_new)
        assert diff["added"] == {
            "garden_area_m2": "Float64"
        }, "an added column must be reported with its dtype"
        assert diff["removed"] == {
            "in_hn_zone": "Boolean"
        }, "a removed column must be reported with its dtype"

    def test_reports_dtype_changes_as_old_new_pairs(self, df_old):
        """A column changing dtype between versions is reported old -> new."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Utf8))
        diff = compare_versions.generate_dict_schema_diff(df_old, df_new)
        assert diff["dtype_changed"] == {
            "UPRN": ("Int64", "String")
        }, "a retyped column must be reported as an (old, new) dtype pair"


class TestGenerateDictUprnChurn:
    """Tests for `generate_dict_uprn_churn`."""

    def test_identical_versions_have_no_churn(self, df_old, df_new_identical):
        """No drift means every UPRN is retained."""
        assert compare_versions.generate_dict_uprn_churn(df_old, df_new_identical) == {
            "n_added": 0,
            "n_removed": 0,
            "n_retained": 4,
            "removed_share": 0.0,
            "n_null_old": 0,
            "n_null_new": 0,
        }, "identical versions must show full retention and zero churn"

    def test_counts_added_removed_and_retained_uprns(self, df_old, df_new_churned):
        """Added/removed/retained are set differences on the UPRN key, and
        removed_share is the fraction of old UPRNs lost."""
        assert compare_versions.generate_dict_uprn_churn(df_old, df_new_churned) == {
            "n_added": 2,
            "n_removed": 1,
            "n_retained": 3,
            "removed_share": 0.25,
            "n_null_old": 0,
            "n_null_new": 0,
        }, "churn must be the set differences on the UPRN key"

    def test_returns_none_without_uprn_column(self, df_old):
        """Outputs without a UPRN column cannot be churn-checked."""
        no_uprn = df_old.drop("UPRN")
        assert (
            compare_versions.generate_dict_uprn_churn(no_uprn, no_uprn) is None
        ), "churn cannot be computed without a UPRN key, so None is expected"

    def test_matches_uprns_across_dtype_changes(self, df_old):
        """A UPRN dtype change between versions must not read as full churn."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Utf8))
        churn = compare_versions.generate_dict_uprn_churn(df_old, df_new)
        assert churn["n_retained"] == 4, "a dtype change must not read as churn"
        assert churn["n_removed"] == 0, "no UPRNs were removed, only retyped"

    def test_matches_uprns_across_int_and_float_dtypes(self, df_old):
        """An Int64-vs-Float64 UPRN mismatch (e.g. a pandas null-upcast) must
        not read as full churn: casting straight to Utf8 would compare "123"
        against "123.0" and miss every match."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Float64))
        churn = compare_versions.generate_dict_uprn_churn(df_old, df_new)
        assert churn["n_retained"] == 4, "int and float UPRN keys must still match"
        assert churn["n_removed"] == 0, "a numeric mismatch must not read as loss"

    def test_null_uprns_are_counted_not_collapsed(self, df_old):
        """Any number of null UPRNs would collapse to one set element and
        silently undercount churn; they are excluded from the churn sets and
        reported as per-version counts instead."""
        df_old_nulled = pl.concat(
            [df_old, pl.DataFrame({"UPRN": [None, None]}, schema={"UPRN": pl.Int64})],
            how="diagonal",
        )
        df_new = df_old.head(2)
        churn = compare_versions.generate_dict_uprn_churn(df_old_nulled, df_new)
        assert (
            churn["n_null_old"] == 2 and churn["n_null_new"] == 0
        ), "each version's null UPRNs must be counted individually"
        assert (
            churn["n_retained"] == 2 and churn["n_removed"] == 2
        ), "null UPRNs must not appear in the churn sets as a phantom member"


class TestGenerateStrChurnNote:
    """Tests for `generate_str_churn_note`."""

    def test_expected_churn_within_tolerance_gets_no_note(self):
        """Churn at or below the rubric's tolerance is expected: no warning."""
        churn = {"n_added": 2, "n_removed": 1, "n_retained": 3, "removed_share": 0.25}
        assert (
            compare_versions.generate_str_churn_note(churn, max_removed_share=0.25)
            is None
        ), "churn within tolerance is expected and must not warn"

    def test_unexpected_uprn_loss_above_tolerance_gets_warning(self):
        """UPRN loss above the rubric's tolerance produces a warning naming
        the observed share and the tolerance."""
        churn = {"n_added": 0, "n_removed": 3, "n_retained": 1, "removed_share": 0.75}
        note = compare_versions.generate_str_churn_note(churn, max_removed_share=0.05)
        assert "75.0%" in note, "the warning must name the observed removed share"
        assert "5.0%" in note, "the warning must name the tolerance it breached"


class TestGetDictTolerances:
    """Tests for `get_dict_tolerances`."""

    @pytest.mark.parametrize("trigger", ["methodology_change", "input_release"])
    def test_each_trigger_rubric_has_a_removed_uprn_tolerance(self, trigger):
        """Both rubrics are configured in base.yaml with the churn tolerance."""
        tolerances = compare_versions.get_dict_tolerances(trigger)
        assert isinstance(
            tolerances["max_removed_uprn_share"], float
        ), "each rubric must configure a numeric removed-UPRN tolerance"

    def test_unknown_trigger_raises_keyerror(self):
        """A trigger without a configured rubric fails loudly."""
        with pytest.raises(KeyError):
            compare_versions.get_dict_tolerances("vibes")


class TestGenerateDfTechTransitions:
    """Tests for `generate_df_tech_transitions`."""

    def test_identical_versions_only_have_diagonal_transitions(
        self, df_old, df_new_identical
    ):
        """No drift means every UPRN keeps its tech (diagonal matrix only)."""
        transitions = compare_versions.generate_df_tech_transitions(
            df_old, df_new_identical
        )
        assert transitions.filter(
            pl.col("assigned_tech_old") != pl.col("assigned_tech_new")
        ).is_empty(), "no drift means no off-diagonal transitions"
        assert (
            transitions["n_uprns"].sum() == 4
        ), "every retained UPRN must appear exactly once in the matrix"

    def test_counts_transitions_for_retained_uprns_only(self, df_old, df_new_churned):
        """Transitions are counted over UPRNs present in both versions; UPRN 3's
        move from District heat network to Individual solution appears."""
        transitions = compare_versions.generate_df_tech_transitions(
            df_old, df_new_churned
        )
        moved = transitions.filter(
            (pl.col("assigned_tech_old") == "District heat network")
            & (pl.col("assigned_tech_new") == "Individual solution")
        )
        assert moved["n_uprns"].to_list() == [1], "UPRN 3's tech move must be counted"
        # Only the 3 retained UPRNs are counted; added/removed ones are churn.
        assert (
            transitions["n_uprns"].sum() == 3
        ), "added/removed UPRNs are churn, not transitions"

    def test_null_tech_assignments_get_a_readable_label(self, df_old):
        """Real outputs contain UPRNs with a null tech (e.g. no building ID);
        they transition as "(null)" rather than an inconsistent None/null mix."""
        df_new = df_old.with_columns(
            pl.when(pl.col("UPRN") == 1)
            .then(None)
            .otherwise(pl.col("assigned_tech"))
            .alias("assigned_tech")
        )
        transitions = compare_versions.generate_df_tech_transitions(df_old, df_new)
        nulled = transitions.filter(pl.col("assigned_tech_new") == "(null)")
        assert nulled["assigned_tech_old"].to_list() == [
            "Individual solution"
        ], "the nulled UPRN's transition must appear under the (null) label"
        assert (
            transitions["assigned_tech_new"].null_count() == 0
        ), "null techs must be labelled, not left as raw nulls"

    def test_returns_none_without_tech_column(self, df_old, df_new_identical):
        """A version missing `assigned_tech` must degrade to None rather than
        raise `ColumnNotFoundError` — this data function must be as safe to
        call directly as its `generate_dict_*` siblings."""
        df_new = df_new_identical.drop("assigned_tech")
        assert (
            compare_versions.generate_df_tech_transitions(df_old, df_new) is None
        ), "a missing tech column must degrade to None, not raise"

    def test_duplicate_uprns_do_not_inflate_transition_counts(self, df_old):
        """A duplicated UPRN in one version must not cross-product into an
        inflated transition count; each version is deduplicated on UPRN
        first."""
        df_new = pl.concat([df_old, df_old.head(1)])  # UPRN 1 appears twice
        transitions = compare_versions.generate_df_tech_transitions(df_old, df_new)
        assert (
            transitions["n_uprns"].sum() == 4
        ), "a duplicated UPRN must not cross-product into extra transitions"

    def test_matches_uprns_across_int_and_float_dtypes(self, df_old):
        """An Int64-vs-Float64 UPRN mismatch must not read as churn here
        either, since the transition matrix joins on the same UPRN key."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Float64))
        transitions = compare_versions.generate_df_tech_transitions(df_old, df_new)
        assert (
            transitions["n_uprns"].sum() == 4
        ), "int and float UPRN keys must still join as the same UPRNs"


class TestGenerateDfTechCounts:
    """Tests for `generate_df_tech_counts`."""

    def test_identical_versions_have_zero_deltas(self, df_old, df_new_identical):
        """No drift means every tech keeps its count."""
        counts = compare_versions.generate_df_tech_counts(df_old, df_new_identical)
        assert (
            counts["n_delta"].abs().sum() == 0
        ), "identical versions must have a zero delta for every tech"

    def test_counts_cover_techs_present_in_only_one_version(
        self, df_old, df_new_churned
    ):
        """A tech present in only one version counts 0 in the other, so
        appearing and disappearing techs both stay visible."""
        counts = compare_versions.generate_df_tech_counts(df_old, df_new_churned)
        by_tech = {row["assigned_tech"]: row for row in counts.iter_rows(named=True)}
        assert by_tech["District heat network"]["n_new"] == 0, (
            "a tech dropped in the new version must count 0 there, "
            "not vanish from the table"
        )
        assert by_tech["Networked heat pump"]["n_old"] == 0, (
            "a tech new in the new version must count 0 in the old, "
            "not vanish from the table"
        )
        assert (
            by_tech["Individual solution"]["n_delta"] == 1
        ), "Individual solution grew 2 -> 3 so its delta must be +1"

    def test_counts_are_per_version_tallies_not_per_retained_uprn(
        self, df_old, df_new_churned
    ):
        """Unlike the transition matrix, counts cover every row of each
        version - added and removed UPRNs included."""
        counts = compare_versions.generate_df_tech_counts(df_old, df_new_churned)
        assert (
            counts["n_old"].sum() == 4 and counts["n_new"].sum() == 5
        ), "tech counts must tally all rows of each version, not the retained join"

    def test_returns_none_without_tech_column(self, df_old):
        """A version missing `assigned_tech` must degrade to None rather than
        raise, like its data-function siblings."""
        no_tech = df_old.drop("assigned_tech")
        assert (
            compare_versions.generate_df_tech_counts(df_old, no_tech) is None
        ), "a missing tech column must return None, not raise"


class TestLoadDictManifest:
    """Tests for `load_dict_manifest`."""

    def test_loads_the_outputs_colocated_manifest(self, mocker):
        """The manifest is read from the output's co-located .manifest.json."""
        manifest = {"stage": "decision_tree", "git_commit": "a" * 40}
        opened = mocker.patch(
            "fsspec.open", mocker.mock_open(read_data=json.dumps(manifest))
        )
        loaded = compare_versions.load_dict_manifest("s3://bucket/dir/output.parquet")
        assert loaded == manifest, "the manifest JSON must round-trip unchanged"
        assert (
            opened.call_args.args[0] == "s3://bucket/dir/output.manifest.json"
        ), "the manifest path must be derived from the output path"

    def test_missing_manifest_returns_none(self, mocker):
        """Pre-#440 outputs have no manifest: degrade to None, don't raise."""
        mocker.patch("fsspec.open", side_effect=FileNotFoundError("no such key"))
        assert (
            compare_versions.load_dict_manifest("s3://bucket/dir/output.parquet")
            is None
        ), "a missing manifest must degrade to None, not raise"


class TestGenerateDictInputVersionChanges:
    """Tests for `generate_dict_input_version_changes`."""

    def test_identical_input_versions_have_empty_changes(self):
        """No input re-release means no changed, added or removed inputs."""
        versions = {"epc.domestic": "s3://bucket/inputs/2026Q1_epc.parquet"}
        assert compare_versions.generate_dict_input_version_changes(
            {"input_versions": versions}, {"input_versions": versions}
        ) == {
            "changed": {},
            "added": {},
            "removed": {},
        }, "identical input versions must produce an empty diff"

    def test_reports_changed_added_and_removed_inputs(self):
        """Input path changes are reported old -> new, alongside inputs only
        one version's manifest records."""
        old = {
            "input_versions": {
                "epc.domestic": "s3://bucket/inputs/2026Q1_epc.parquet",
                "geodata.gb_code_points": "s3://bucket/inputs/2025_codepoint.zip",
            }
        }
        new = {
            "input_versions": {
                "epc.domestic": "s3://bucket/inputs/2026Q2_epc.parquet",
                "geodata.uk_osopen_uprn": "s3://bucket/inputs/2026_uprn.zip",
            }
        }
        assert compare_versions.generate_dict_input_version_changes(old, new) == {
            "changed": {
                "epc.domestic": (
                    "s3://bucket/inputs/2026Q1_epc.parquet",
                    "s3://bucket/inputs/2026Q2_epc.parquet",
                )
            },
            "added": {"geodata.uk_osopen_uprn": "s3://bucket/inputs/2026_uprn.zip"},
            "removed": {
                "geodata.gb_code_points": "s3://bucket/inputs/2025_codepoint.zip"
            },
        }, "changed inputs pair old and new paths; one-sided ones are added/removed"


class TestStageModulePaths:
    """Tests for the curated `STAGE_MODULE_PATHS` lists."""

    def test_covers_the_same_stages_as_the_run_manifest(self):
        """Commit-log scoping mirrors the run manifest's curated stage list."""
        assert set(compare_versions.STAGE_MODULE_PATHS) == set(
            manifest_utils.STAGE_INPUT_KEYS
        ), "commit-log scoping must cover exactly the manifest's curated stages"

    def test_every_curated_path_exists_in_the_repo(self):
        """A renamed module must break this test, not silently empty the log."""
        for stage, paths in compare_versions.STAGE_MODULE_PATHS.items():
            for path in paths:
                assert (
                    PROJECT_DIR / path
                ).exists(), f"Missing module path for stage {stage}: {path}"


class TestGenerateListCommitLog:
    """Tests for `generate_list_commit_log`."""

    def test_scopes_git_log_to_the_stages_module_paths(self, mocker):
        """git log runs over old..new restricted to the stage's curated
        paths, after confirming old is an ancestor of new."""
        run = mocker.patch(
            "subprocess.run",
            side_effect=[
                subprocess.CompletedProcess(args=[], returncode=0),  # merge-base
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="abc1234 Fix tree\ndef5678 Tune zones\n",
                ),
            ],
        )
        commits = compare_versions.generate_list_commit_log(
            "a" * 40, "b" * 40, "decision_tree"
        )
        assert commits == [
            "abc1234 Fix tree",
            "def5678 Tune zones",
        ], "the scoped git log's lines must come back as the commit list"
        command = run.call_args.args[0]
        assert (
            f"{'a' * 40}..{'b' * 40}" in command
        ), "git log must be scoped to the old..new commit range"
        paths = command[command.index("--") + 1 :]
        assert (
            paths == compare_versions.STAGE_MODULE_PATHS["decision_tree"]
        ), "git log must be restricted to the stage's curated module paths"

    def test_same_commit_returns_empty_log_without_running_git(self, mocker):
        """Two outputs from the same commit cannot differ by code: empty log."""
        run = mocker.patch("subprocess.run")
        assert (
            compare_versions.generate_list_commit_log("a" * 40, "a" * 40, "uprns") == []
        ), "the same commit on both sides cannot differ by code"
        run.assert_not_called()

    def test_unknown_recorded_commit_returns_none(self, mocker):
        """A manifest recording the "unknown" commit sentinel cannot be scoped."""
        run = mocker.patch("subprocess.run")
        assert (
            compare_versions.generate_list_commit_log("unknown", "b" * 40, "uprns")
            is None
        ), "an unrecorded commit cannot scope a log, so None is expected"
        run.assert_not_called()

    def test_commit_old_not_an_ancestor_of_commit_new_returns_none(self, mocker):
        """`old..new` silently omits commits when old isn't an ancestor of
        new (e.g. old came from a since-rebased branch); this must degrade
        to None rather than return an incomplete log, and must not run
        `git log` at all once the ancestor check has failed."""
        run = mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git merge-base"),
        )
        assert (
            compare_versions.generate_list_commit_log("a" * 40, "b" * 40, "uprns")
            is None
        ), "a non-ancestor old commit must degrade to None, not an incomplete log"
        run.assert_called_once()

    def test_commits_missing_from_local_history_return_none(self, mocker):
        """git log itself failing after a successful ancestor check (e.g. a
        shallow clone missing older history) degrades to None."""
        run = mocker.patch(
            "subprocess.run",
            side_effect=[
                subprocess.CompletedProcess(args=[], returncode=0),  # merge-base
                subprocess.CalledProcessError(128, "git log"),
            ],
        )
        assert (
            compare_versions.generate_list_commit_log("a" * 40, "b" * 40, "uprns")
            is None
        ), "git log failing after the ancestor check must degrade to None"
        assert (
            run.call_count == 2
        ), "both the ancestor check and git log must have been attempted"


class TestGetStrStageOutputPath:
    """Tests for `get_str_stage_output_path`."""

    def test_resolves_the_decision_tree_uprn_level_output(self):
        """The decision-tree comparison reads the UPRN-level output, the
        stable join key the churn check and transition matrix rely on."""
        assert compare_versions.get_str_stage_output_path(
            "decision_tree", "plymouth", "20260722"
        ) == (
            "s3://asf-local-heat-planning-tool/outputs/data/plymouth/20260722/"
            "plymouth_uprns_most_suitable_tech.parquet"
        ), "the decision-tree comparison must read the UPRN-level output"

    def test_fills_the_clustering_tolerance_placeholder(self):
        """The contextual-features template's {tolerance_m} placeholder is
        filled from config so the path matches what the pipeline saved."""
        tolerance_m = config["constant"]["clustering"]["tolerance_m"]
        path = compare_versions.get_str_stage_output_path(
            "compute_contextual_features", "plymouth", "20260722"
        )
        assert path.endswith(
            f"_clusters_contextual_features_{tolerance_m}m.geojson"
        ), "the tolerance placeholder must be filled from config"

    def test_covers_the_same_stages_as_the_run_manifest(self):
        """Output-dataset mapping mirrors the run manifest's curated stages."""
        assert set(compare_versions.STAGE_OUTPUT_DATASETS) == set(
            manifest_utils.STAGE_INPUT_KEYS
        ), "output resolution must cover exactly the manifest's curated stages"

    def test_resolves_a_version_saved_under_a_previous_tolerance(self, mocker):
        """A contextual-features version saved before a tolerance_m config
        change must still resolve (via glob) so past releases stay
        comparable across the very methodology changes the tool exists for."""
        mocker.patch.object(
            compare_versions.save_utils,
            "get_str_output_path",
            side_effect=FileNotFoundError("No file found"),
        )
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = [
            "bucket/outputs/data/plymouth/20260601/"
            "plymouth_clusters_contextual_features_10m.geojson"
        ]
        path = compare_versions.get_str_stage_output_path(
            "compute_contextual_features", "plymouth", "20260601", check_exists=True
        )
        assert path == (
            "s3://bucket/outputs/data/plymouth/20260601/"
            "plymouth_clusters_contextual_features_10m.geojson"
        ), "a single glob match under another tolerance must be resolved and returned"

    def test_missing_version_still_raises_when_no_tolerance_variant_exists(
        self, mocker
    ):
        """The tolerance-glob fallback must not mask a genuinely missing
        version: zero (or ambiguous) matches re-raise the original error."""
        mocker.patch.object(
            compare_versions.save_utils,
            "get_str_output_path",
            side_effect=FileNotFoundError("No file found"),
        )
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = []
        with pytest.raises(FileNotFoundError):
            compare_versions.get_str_stage_output_path(
                "compute_contextual_features",
                "plymouth",
                "20260601",
                check_exists=True,
            )


class TestGenerateListReleaseDates:
    """Tests for `generate_list_release_dates`."""

    def test_lists_dated_versions_sorted_oldest_first(self, mocker):
        """Release dates come back sorted oldest first, whatever order S3
        lists them in (YYYYMMDD sorts chronologically as strings)."""
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = [
            "bucket/outputs/data/plymouth/20260722/plymouth_uprns_most_suitable_tech.parquet",
            "bucket/outputs/data/plymouth/20260601/plymouth_uprns_most_suitable_tech.parquet",
        ]
        dates = compare_versions.generate_list_release_dates(
            "decision_tree", "plymouth"
        )
        assert dates == [
            "20260601",
            "20260722",
        ], "release dates must be sorted oldest first regardless of S3 list order"

    def test_skips_directories_that_are_not_release_dates(self, mocker):
        """A stray non-date directory (e.g. a manual 'latest' copy) must be
        skipped, not returned as a version or crash date parsing."""
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = [
            "bucket/outputs/data/plymouth/latest/plymouth_uprns_most_suitable_tech.parquet",
            "bucket/outputs/data/plymouth/20260722/plymouth_uprns_most_suitable_tech.parquet",
        ]
        dates = compare_versions.generate_list_release_dates(
            "decision_tree", "plymouth"
        )
        assert dates == [
            "20260722"
        ], "non-date directories must be skipped, not listed as versions"

    def test_glob_pattern_fixes_stage_and_local_authority(self, mocker):
        """The S3 glob fixes stage and LA and wildcards the dated directory,
        so another LA's versions can't leak into the list."""
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = []
        compare_versions.generate_list_release_dates("decision_tree", "plymouth")
        pattern = fs.glob.call_args.args[0]
        assert pattern.endswith(
            "/plymouth/*/plymouth_uprns_most_suitable_tech.parquet"
        ), "stage filename and LA must stay fixed; the dated segment is the wildcard"

    def test_glob_pattern_wildcards_the_clustering_tolerance(self, mocker):
        """The contextual-features filename embeds the clustering tolerance;
        discovery must wildcard it so versions saved under a previous
        tolerance are still found."""
        fs = mocker.patch("s3fs.S3FileSystem").return_value
        fs.glob.return_value = []
        compare_versions.generate_list_release_dates(
            "compute_contextual_features", "plymouth"
        )
        pattern = fs.glob.call_args.args[0]
        assert pattern.endswith(
            "_clusters_contextual_features_*m.geojson"
        ), "the tolerance segment must be a wildcard, not the current config value"


class TestGetTupleDefaultReleaseDates:
    """Tests for `get_tuple_default_release_dates`."""

    def test_picks_the_latest_two_versions(self, mocker):
        """With more than two versions available, the default comparison is
        the latest against the one before it."""
        mocker.patch.object(
            compare_versions,
            "generate_list_release_dates",
            return_value=["20260601", "20260708", "20260722"],
        )
        assert compare_versions.get_tuple_default_release_dates(
            "decision_tree", "plymouth"
        ) == (
            "20260708",
            "20260722",
        ), "the default comparison must be the latest two versions, (older, newer)"

    def test_fewer_than_two_versions_raises(self, mocker):
        """One or zero dated versions cannot be compared: fail loudly with
        the explicit-dates escape hatch rather than comparing a version to
        itself."""
        mocker.patch.object(
            compare_versions,
            "generate_list_release_dates",
            return_value=["20260722"],
        )
        with pytest.raises(FileNotFoundError, match="explicitly"):
            compare_versions.get_tuple_default_release_dates(
                "decision_tree", "plymouth"
            )


@pytest.fixture()
def manifests():
    """Old/new run manifests with distinct commits and one input re-release."""
    return (
        {
            "stage": "decision_tree",
            "git_commit": "a" * 40,
            "input_versions": {"epc.domestic": "s3://bucket/inputs/2026Q1_epc.parquet"},
        },
        {
            "stage": "decision_tree",
            "git_commit": "b" * 40,
            "input_versions": {"epc.domestic": "s3://bucket/inputs/2026Q2_epc.parquet"},
        },
    )


def generate_report(df_old, df_new, manifest_old, manifest_new, **overrides):
    """Build a report with representative defaults, overridable per test."""
    kwargs = {
        "stage": "decision_tree",
        "local_authority": "plymouth",
        "trigger": "methodology_change",
        "release_date_old": "20260601",
        "release_date_new": "20260722",
        "path_old": "s3://bucket/outputs/plymouth/20260601/old.parquet",
        "path_new": "s3://bucket/outputs/plymouth/20260722/new.parquet",
    }
    kwargs.update(overrides)
    return compare_versions.generate_str_report(
        df_old, df_new, manifest_old, manifest_new, **kwargs
    )


class TestGenerateStrReport:
    """Tests for `generate_str_report`."""

    def test_states_the_trigger_rubric_and_versions(
        self, df_old, df_new_identical, manifests, mocker
    ):
        """The report names both versions and the rubric it was read against."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        report = generate_report(df_old, df_new_identical, *manifests)
        assert "methodology_change" in report, "the report must name its rubric"
        assert "20260601" in report, "the report must name the old version"
        assert "20260722" in report, "the report must name the new version"

    def test_covers_counts_schema_churn_and_input_changes(
        self, df_old, df_new_churned, manifests, mocker
    ):
        """Count delta, schema diff, UPRN churn and manifest-recorded input
        changes each get a report section."""
        mocker.patch.object(
            compare_versions,
            "generate_list_commit_log",
            return_value=["abc1234 Tune decision tree"],
        )
        report = generate_report(df_old, df_new_churned, *manifests)
        assert "Row and UPRN counts" in report, "the count-delta section must appear"
        assert "Schema diff" in report, "the schema-diff section must appear"
        assert "UPRN churn" in report, "the churn section must appear"
        assert "2026Q2_epc" in report, "the re-released input must be named"
        assert (
            "abc1234 Tune decision tree" in report
        ), "the scoped commit log must appear in the report"

    def test_transition_matrix_only_for_the_decision_tree_stage(
        self, df_old, df_new_churned, manifests, mocker
    ):
        """The tech transition matrix is UPRN-level, so only the decision-tree
        stage's report includes it."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        decision_tree = generate_report(df_old, df_new_churned, *manifests)
        other = generate_report(
            df_old, df_new_churned, *manifests, stage="add_features"
        )
        assert (
            "Tech-assignment transitions" in decision_tree
        ), "the decision-tree report must carry the transition matrix"
        assert (
            "Tech-assignment transitions" not in other
        ), "a UPRN-level matrix must not appear for other stages"

    def test_missing_manifests_name_both_versions_in_the_note(
        self, df_old, df_new_identical
    ):
        """When both versions lack a manifest, the note must not attribute
        the gap to only one of them."""
        report = generate_report(df_old, df_new_identical, None, None)
        assert (
            "manifest missing for the old and new versions" in report.lower()
        ), "the note must attribute the missing manifests to both versions"
        assert (
            "Row and UPRN counts" in report
        ), "data sections must still render without manifests"

    def test_missing_manifest_names_only_the_version_that_lacks_one(
        self, df_old, df_new_identical, manifests
    ):
        """When only one version lacks a manifest, the note names that
        version specifically, not both."""
        _, manifest_new = manifests
        report = generate_report(df_old, df_new_identical, None, manifest_new)
        assert (
            "manifest missing for the old version " in report.lower()
        ), "the note must name the one version lacking a manifest"
        assert (
            "old and new" not in report.lower()
        ), "the note must not blame both versions when only one lacks a manifest"

    def test_transition_matrix_skips_when_tech_column_missing(
        self, df_old, manifests, mocker
    ):
        """A version missing `assigned_tech` (e.g. dropped by a schema
        change) must not crash report generation; the schema diff already
        surfaces the drop."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.drop("assigned_tech")
        report = generate_report(df_old, df_new, *manifests)
        assert "Skipped" in report, "the matrix must skip, not crash the report"
        assert "assigned_tech" in report, "the skip note must name the missing column"

    def test_transition_matrix_skips_when_no_uprns_retained(
        self, df_old, manifests, mocker
    ):
        """Total UPRN churn (no overlap between versions) must render a note
        rather than an empty, malformed markdown table."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.with_columns((pl.col("UPRN") + 100).alias("UPRN"))
        report = generate_report(df_old, df_new, *manifests)
        assert (
            "No UPRNs retained across versions; matrix skipped." in report
        ), "total churn must render a note, not a malformed table"

    def test_transition_matrix_survives_a_colliding_tech_label(
        self, df_old, manifests, mocker
    ):
        """A real tech label equal to the pivot's own index column name
        ("assigned_tech_old") must not crash the matrix — the pivot renames
        its index to an internal column first."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.with_columns(
            pl.when(pl.col("UPRN") == 1)
            .then(pl.lit("assigned_tech_old"))
            .otherwise(pl.col("assigned_tech"))
            .alias("assigned_tech")
        )
        report = generate_report(df_old, df_new, *manifests)
        assert "assigned_tech_old" in report, "a colliding label must still render"

    def test_unexpected_uprn_loss_warning_appears(self, df_old, manifests, mocker):
        """UPRN loss above the rubric tolerance surfaces as a warning line."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.head(1)
        report = generate_report(df_old, df_new, *manifests)
        assert "WARNING" in report, "above-tolerance UPRN loss must surface a warning"

    def test_omitted_trigger_reports_raw_numbers_without_rubric(
        self, df_old, manifests, mocker
    ):
        """With no trigger there is no rubric to read against: the report
        must carry no tolerance warnings even for churn that would breach
        every configured rubric, and must say the trigger was not supplied."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.head(1)  # 75% UPRN loss breaches both rubrics
        report = generate_report(df_old, df_new, *manifests, trigger=None)
        assert (
            "WARNING" not in report
        ), "no trigger means no rubric, so no tolerance warnings may appear"
        assert (
            "read against" not in report
        ), "the report must not claim a rubric was applied when none was supplied"
        assert (
            "not supplied" in report
        ), "the report must state that no trigger was supplied"

    def test_marginal_tech_counts_appear_for_both_output_levels(
        self, df_old, df_new_churned, manifests, mocker
    ):
        """The decision-tree report carries per-tech counts for the
        UPRN-level output and the building-level output."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        report = generate_report(
            df_old,
            df_new_churned,
            *manifests,
            df_buildings_old=df_old.head(2),
            df_buildings_new=df_new_churned.head(3),
        )
        assert (
            "Per-tech counts (UPRN-level)" in report
        ), "the decision-tree report must include UPRN-level per-tech counts"
        assert (
            "Per-tech counts (building-level)" in report
        ), "the decision-tree report must include building-level per-tech counts"

    def test_building_counts_skip_when_building_output_missing(
        self, df_old, df_new_churned, manifests, mocker
    ):
        """A version without a building-level output (or one that failed to
        load) degrades the building counts to a note, not a crash."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        report = generate_report(df_old, df_new_churned, *manifests)
        assert (
            "Per-tech counts (building-level)" in report
            and "Skipped: output missing" in report
        ), "missing building output must be noted in its section, not crash"

    def test_tech_counts_only_for_the_decision_tree_stage(
        self, df_old, df_new_churned, manifests, mocker
    ):
        """Per-tech counts read the decision-tree outputs, so other stages'
        reports must not carry the sections."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        report = generate_report(
            df_old, df_new_churned, *manifests, stage="add_features"
        )
        assert (
            "Per-tech counts" not in report
        ), "non-decision-tree stages must not carry per-tech count sections"


class TestGenerateStrReportGeometrySections:
    """Tests for `generate_str_report`'s cluster geometry sections."""

    def test_geometry_drift_surfaces_in_counts_and_area(
        self, df_clusters_old, df_clusters_merged, df_areas_old, df_areas_merged
    ):
        """A cluster merge (genuine geometry drift invisible to the tabular
        checks) must surface as a cluster count delta and an area delta,
        with the CRS and units stated."""
        report = generate_report(
            df_clusters_old,
            df_clusters_merged,
            None,
            None,
            stage="cluster",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_merged,
        )
        assert "Cluster geometry" in report, "the geometry section must appear"
        assert (
            "| Clusters | 3 | 2 | -1 |" in report
        ), "the cluster merge must appear as a -1 cluster count delta"
        assert (
            "| Total area (m²) | 300.0 | 310.0 | +10.0 |" in report
        ), "the absorbed 10 m² gap must appear as the total-area delta"
        assert "EPSG:27700" in report, "the report must state the CRS areas use"

    def test_stable_versions_show_zero_geometry_deltas(
        self, df_clusters_old, df_areas_old
    ):
        """No geometry drift means zero cluster and area deltas."""
        report = generate_report(
            df_clusters_old,
            df_clusters_old.clone(),
            None,
            None,
            stage="cluster",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_old.clone(),
        )
        assert (
            "| Clusters | 3 | 3 | +0 |" in report
        ), "stable versions must show a zero cluster count delta"
        assert (
            "| Total area (m²) | 300.0 | 300.0 | +0.0 |" in report
        ), "stable versions must show a zero total-area delta"

    def test_simplified_geometry_caveat_only_at_the_contextual_stage(
        self, df_clusters_old, df_areas_old
    ):
        """The contextual-features geojson carries simplified geometry, so
        its report warns that area differences may be simplification
        artefacts; the cluster stage's exact geometry needs no caveat."""
        kwargs = dict(
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_old.clone(),
        )
        contextual = generate_report(
            df_clusters_old,
            df_clusters_old,
            None,
            None,
            stage="compute_contextual_features",
            **kwargs,
        )
        cluster = generate_report(
            df_clusters_old, df_clusters_old, None, None, stage="cluster", **kwargs
        )
        assert (
            "simplified geometry" in contextual
        ), "the contextual-features report must carry the simplified-geometry caveat"
        assert (
            "simplified geometry" not in cluster
        ), "the cluster stage's exact geometry must not carry the caveat"

    def test_distribution_sections_report_the_statistics_per_version(
        self, df_clusters_old, df_areas_old, df_areas_merged
    ):
        """Each distribution gets a section with Q1, Q3, min, max and mean
        for both versions; the contextual stage also covers n_UPRNs."""
        df_old = df_clusters_old.with_columns(pl.lit(5).alias("n_UPRNs"))
        df_new = df_old.with_columns(pl.lit(8).alias("n_UPRNs"))
        report = generate_report(
            df_old,
            df_new,
            None,
            None,
            stage="compute_contextual_features",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_merged,
        )
        assert (
            "Distribution: area_m2" in report
        ), "the cluster-area distribution section must appear"
        assert (
            "Distribution: n_UPRNs" in report
        ), "the UPRNs-per-cluster distribution section must appear"
        for statistic in ("Min", "Q1", "Mean", "Q3", "Max"):
            assert (
                f"| {statistic} |" in report
            ), f"each distribution must report {statistic} per version"
        assert (
            "| Mean | 5.0 | 8.0 |" in report
        ), "the n_UPRNs shift must appear as old and new means"

    def test_plots_are_embedded_as_image_links(
        self, df_clusters_old, df_areas_old, df_areas_merged
    ):
        """Saved distribution plots are embedded in the report via image
        links, next to their distribution's statistics."""
        report = generate_report(
            df_clusters_old,
            df_clusters_old,
            None,
            None,
            stage="cluster",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_merged,
            plot_files={"area_m2": "cluster_plymouth_area_m2.png"},
        )
        assert (
            "![Distribution of area_m2: old vs new](cluster_plymouth_area_m2.png)"
            in report
        ), "the saved plot must be embedded via a markdown image link"

    def test_missing_cluster_id_degrades_the_count_to_a_note(
        self, df_old, df_areas_old
    ):
        """A geometry-stage version without a cluster_id column (schema
        change) must degrade the count to a note, not crash the report."""
        report = generate_report(
            df_old,
            df_old,
            None,
            None,
            stage="cluster",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_old.clone(),
        )
        assert (
            "cluster_id" in report and "Cluster geometry" in report
        ), "the missing cluster_id must be noted inside the geometry section"

    def test_layered_output_scopes_headline_checks_to_the_clusters_layer(
        self, df_clusters_old, df_areas_old
    ):
        """A new-format multi-layer output (ward boundaries bundled with the
        clusters) must not swamp the cluster count and total area: the
        headline checks cover the clusters layer only, the pre-layers old
        version counts entirely as clusters, and the report says so."""
        df_new = pl.DataFrame(
            {
                "cluster_id": ["HP_1", "DHN_1", None],
                "layer": [
                    compare_versions.CLUSTER_LAYER,
                    compare_versions.CLUSTER_LAYER,
                    "ward_boundaries",
                ],
            }
        )
        df_areas_new = pl.DataFrame(
            {
                "area_m2": [210.0, 100.0, 5_000_000.0],
                "layer": [
                    compare_versions.CLUSTER_LAYER,
                    compare_versions.CLUSTER_LAYER,
                    "ward_boundaries",
                ],
            }
        )
        report = generate_report(
            df_clusters_old,
            df_new,
            None,
            None,
            stage="compute_contextual_features",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_new,
        )
        assert (
            "| Clusters | 3 | 2 | -1 |" in report
        ), "the ward row (null cluster_id) must not count as a cluster"
        assert (
            "| Total area (m²) | 300.0 | 310.0 | +10.0 |" in report
        ), "the 5M m² ward polygon must not enter the total-area check"
        assert (
            f"`{compare_versions.CLUSTER_LAYER}` layer only" in report
        ), "the report must state the headline checks cover the clusters layer only"

    def test_unlayered_versions_carry_no_layer_filtering_note(
        self, df_clusters_old, df_areas_old
    ):
        """Two pre-layers versions have nothing filtered, so the report
        must not claim a layer scope that was never applied."""
        report = generate_report(
            df_clusters_old,
            df_clusters_old.clone(),
            None,
            None,
            stage="cluster",
            trigger=None,
            df_areas_old=df_areas_old,
            df_areas_new=df_areas_old.clone(),
        )
        assert (
            "layer only" not in report
        ), "no layer column anywhere means no filtering note may appear"

    def test_geometry_sections_only_for_geometry_stages(
        self, df_old, df_new_identical, manifests, mocker
    ):
        """Stages without cluster geometry must not carry geometry or
        distribution sections."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        report = generate_report(df_old, df_new_identical, *manifests)
        assert (
            "Cluster geometry" not in report and "Distribution:" not in report
        ), "non-geometry stages must not carry geometry sections"


class TestLoadTransformDfStageOutput:
    """Tests for `load_transform_df_stage_output`."""

    def test_loads_geoparquet_without_the_geometry_column(self, tmp_path):
        """Geopandas-written outputs carry geoarrow extension columns polars
        cannot read; the loader drops them and returns the tabular columns."""
        import geopandas as gpd
        from shapely.geometry import Point

        path = tmp_path / "output.parquet"
        gpd.GeoDataFrame(
            {"UPRN": [1, 2], "assigned_tech": ["Individual solution"] * 2},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:27700",
        ).to_parquet(path)
        df = compare_versions.load_transform_df_stage_output(str(path))
        assert "geometry" not in df.columns, "geoarrow columns must be dropped"
        assert df["UPRN"].to_list() == [1, 2], "tabular columns must survive the drop"

    def test_loads_plain_parquet_unchanged(self, tmp_path):
        """Polars-written outputs have no geometry and load as-is."""
        path = tmp_path / "output.parquet"
        pl.DataFrame({"UPRN": [1, 2], "in_hn_zone": [True, False]}).write_parquet(path)
        df = compare_versions.load_transform_df_stage_output(str(path))
        assert df.columns == ["UPRN", "in_hn_zone"], "plain parquet columns load as-is"
        assert df.height == 2, "plain parquet rows must load unchanged"

    def test_zero_feature_geojson_degrades_to_an_empty_dataframe(self, mocker):
        """A geojson with every feature filtered out (e.g. compute_contextual
        features' documented empty-cluster temporary fix) must not crash the
        comparison; it degrades to an empty output instead."""
        mocker.patch(
            "asf_heat_pump_suitability.getters.base_getters.load_gdf_from_s3_geojson",
            side_effect=ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column "
                "is not supported"
            ),
        )
        df = compare_versions.load_transform_df_stage_output(
            "s3://bucket/dir/output.geojson"
        )
        assert df.is_empty(), "a zero-feature geojson must degrade to an empty frame"


class TestLoadDfBuildingsTech:
    """Tests for `load_df_buildings_tech`."""

    def test_fetches_only_the_tech_column(self, tmp_path):
        """The building-level output feeds per-tech counts alone, so only
        the tech-assignment column is downloaded."""
        path = tmp_path / "buildings.parquet"
        pl.DataFrame(
            {"ID": [1, 2], "assigned_tech": ["Individual solution"] * 2}
        ).write_parquet(path)
        df = compare_versions.load_df_buildings_tech(str(path))
        assert df.columns == [
            "assigned_tech"
        ], "only the tech column may be fetched from the building-level output"

    def test_missing_tech_column_loads_as_an_empty_frame(self, tmp_path):
        """A building-level version without the tech column degrades to an
        empty frame, which the counts section reports as a missing column
        rather than crashing on a column-not-found read."""
        path = tmp_path / "buildings.parquet"
        pl.DataFrame({"ID": [1, 2]}).write_parquet(path)
        df = compare_versions.load_df_buildings_tech(str(path))
        assert (
            df.is_empty() and "assigned_tech" not in df.columns
        ), "a missing tech column must load as empty, not raise"


class TestLoadTupleDfBuildings:
    """Tests for `load_tuple_df_buildings`."""

    def test_missing_output_degrades_to_none_pair(self, mocker):
        """A version without a building-level output degrades to (None,
        None) with a warning, not an aborted comparison."""
        mocker.patch.object(
            compare_versions,
            "get_str_buildings_output_path",
            side_effect=FileNotFoundError("No file found at s3://bucket/x.parquet"),
        )
        assert compare_versions.load_tuple_df_buildings(
            "plymouth", "20260601", "20260722"
        ) == (None, None), "a missing building output must degrade to (None, None)"

    def test_unreadable_output_degrades_to_none_pair(self, mocker):
        """A present-but-unreadable building output (e.g. a corrupt parquet
        raising ArrowInvalid) must also degrade, not crash the comparison."""
        import pyarrow

        mocker.patch.object(
            compare_versions,
            "get_str_buildings_output_path",
            return_value="s3://bucket/x.parquet",
        )
        mocker.patch.object(
            compare_versions,
            "load_df_buildings_tech",
            side_effect=pyarrow.ArrowInvalid("corrupt parquet"),
        )
        assert compare_versions.load_tuple_df_buildings(
            "plymouth", "20260601", "20260722"
        ) == (None, None), "an unreadable building output must degrade to (None, None)"

    def test_checks_both_paths_before_downloading_anything(self, mocker):
        """Both paths are existence-checked before any download, so a
        missing new version can't waste a full download of the old one."""
        mocker.patch.object(
            compare_versions,
            "get_str_buildings_output_path",
            side_effect=[
                "s3://bucket/old.parquet",
                FileNotFoundError("No file found at s3://bucket/new.parquet"),
            ],
        )
        load = mocker.patch.object(compare_versions, "load_df_buildings_tech")
        compare_versions.load_tuple_df_buildings("plymouth", "20260601", "20260722")
        load.assert_not_called()


@pytest.fixture(scope="module")
def df_clusters_old():
    """Old-version cluster-level output: three clusters across two techs."""
    return pl.DataFrame(
        {
            "cluster_id": ["HP_1", "HP_2", "DHN_1"],
            "assigned_tech": [
                "Networked heat pump",
                "Networked heat pump",
                "District heat network",
            ],
        }
    )


@pytest.fixture(scope="module")
def df_clusters_merged(df_clusters_old):
    """New-version cluster-level output with genuine geometry drift: the two
    heat-pump clusters merged into one."""
    return pl.DataFrame(
        {
            "cluster_id": ["HP_1", "DHN_1"],
            "assigned_tech": ["Networked heat pump", "District heat network"],
        }
    )


@pytest.fixture(scope="module")
def df_areas_old():
    """Old-version per-cluster areas: three 100 m² clusters."""
    return pl.DataFrame({"area_m2": [100.0, 100.0, 100.0]})


@pytest.fixture(scope="module")
def df_areas_merged():
    """New-version per-cluster areas after the merge: the merged cluster
    absorbed the 10 m² gap between its parents (100 + 100 + 10)."""
    return pl.DataFrame({"area_m2": [210.0, 100.0]})


class TestGenerateDictClusterCountDelta:
    """Tests for `generate_dict_cluster_count_delta`."""

    def test_identical_versions_have_zero_delta(self, df_clusters_old):
        """No drift means no change in the distinct-cluster count."""
        counts = compare_versions.generate_dict_cluster_count_delta(
            df_clusters_old, df_clusters_old.clone()
        )
        assert counts == {
            "clusters_old": 3,
            "clusters_new": 3,
            "clusters_delta": 0,
        }, "identical versions must show no cluster count change"

    def test_merged_clusters_show_a_negative_delta(
        self, df_clusters_old, df_clusters_merged
    ):
        """Two clusters merging into one is genuine geometry drift the
        tabular checks cannot see; the count delta must surface it."""
        counts = compare_versions.generate_dict_cluster_count_delta(
            df_clusters_old, df_clusters_merged
        )
        assert counts == {
            "clusters_old": 3,
            "clusters_new": 2,
            "clusters_delta": -1,
        }, "a cluster merge must appear as a negative cluster count delta"

    def test_counts_distinct_cluster_ids_not_rows(self, df_clusters_old):
        """A cluster-id column with repeated values (e.g. a UPRN-level frame)
        must count distinct clusters, not rows."""
        repeated = pl.concat([df_clusters_old, df_clusters_old])
        counts = compare_versions.generate_dict_cluster_count_delta(
            repeated, df_clusters_old
        )
        assert (
            counts["clusters_old"] == 3
        ), "repeated cluster ids must count as one cluster each"

    def test_returns_none_without_cluster_id_column(self, df_old):
        """Outputs without a cluster_id column (e.g. UPRN-stage outputs)
        cannot be cluster-counted."""
        assert (
            compare_versions.generate_dict_cluster_count_delta(df_old, df_old) is None
        ), "no cluster_id column means no cluster count, so None is expected"


class TestGenerateDictTotalAreaDelta:
    """Tests for `generate_dict_total_area_delta`."""

    def test_identical_versions_have_zero_delta(self, df_areas_old):
        """No drift means no total-area change."""
        totals = compare_versions.generate_dict_total_area_delta(
            df_areas_old, df_areas_old.clone()
        )
        assert totals == {
            "area_m2_old": 300.0,
            "area_m2_new": 300.0,
            "area_m2_delta": 0.0,
        }, "identical versions must show no total-area change"

    def test_reports_the_total_area_delta_in_m2(self, df_areas_old, df_areas_merged):
        """The cluster merge grew the total area by the 10 m² gap it
        absorbed; the delta must surface it in m²."""
        totals = compare_versions.generate_dict_total_area_delta(
            df_areas_old, df_areas_merged
        )
        assert totals == {
            "area_m2_old": 300.0,
            "area_m2_new": 310.0,
            "area_m2_delta": 10.0,
        }, "the total-area delta must be new minus old, in m²"

    def test_a_version_with_no_clusters_totals_zero(self, df_areas_old):
        """An empty areas frame (zero-feature geojson) totals 0 m² rather
        than crashing the delta."""
        empty = pl.DataFrame(schema={"area_m2": pl.Float64})
        totals = compare_versions.generate_dict_total_area_delta(df_areas_old, empty)
        assert (
            totals["area_m2_new"] == 0.0 and totals["area_m2_delta"] == -300.0
        ), "a version with no clusters must total zero area, not crash"


class TestLoadDfClusterAreas:
    """Tests for `load_df_cluster_areas`."""

    @staticmethod
    def gdf_two_squares(crs: str):
        """Two axis-aligned squares of 100 and 400 m², built in EPSG:27700
        near Plymouth and converted to the requested CRS."""
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {"cluster_id": ["HP_1", "HP_2"]},
            geometry=[
                box(250000, 55000, 250010, 55010),
                box(250100, 55100, 250120, 55120),
            ],
            crs="EPSG:27700",
        )
        return gdf.to_crs(crs)

    def test_geoparquet_areas_measured_in_m2(self, tmp_path):
        """The cluster stage's geoparquet carries exact EPSG:27700 geometry;
        areas come back in m², one row per cluster."""
        path = tmp_path / "clusters.parquet"
        self.gdf_two_squares("EPSG:27700").to_parquet(path)
        df = compare_versions.load_df_cluster_areas(str(path))
        assert df.columns == ["area_m2"], "only the derived area column is returned"
        assert df["area_m2"].to_list() == [
            100.0,
            400.0,
        ], "areas must be measured in m² on the saved EPSG:27700 geometry"

    def test_geoparquet_in_another_crs_is_reprojected_before_measuring(self, tmp_path):
        """A geoparquet not in the target CRS must be reprojected to
        EPSG:27700 first — measuring in EPSG:4326 would return degrees²."""
        path = tmp_path / "clusters.parquet"
        self.gdf_two_squares("EPSG:4326").to_parquet(path)
        df = compare_versions.load_df_cluster_areas(str(path))
        assert df["area_m2"].to_list() == pytest.approx(
            [100.0, 400.0], rel=1e-3
        ), "areas must be measured in metres after reprojection, not degrees"

    def test_geojson_is_reprojected_to_the_target_crs_before_measuring(self, mocker):
        """The contextual-features geojson is saved in EPSG:4326; its areas
        must be measured after reprojecting back to EPSG:27700."""
        load = mocker.patch(
            "asf_heat_pump_suitability.getters.base_getters.load_gdf_from_s3_geojson",
            return_value=self.gdf_two_squares("EPSG:4326"),
        )
        df = compare_versions.load_df_cluster_areas("s3://bucket/dir/output.geojson")
        assert load.call_args.kwargs.get("crs") == "EPSG:4326" or (
            "EPSG:4326" in load.call_args.args
        ), "the geojson must be loaded as the EPSG:4326 it is saved in"
        assert df["area_m2"].to_list() == pytest.approx(
            [100.0, 400.0], rel=1e-3
        ), "geojson areas must be measured in m² after reprojection"

    def test_zero_feature_geojson_degrades_to_an_empty_frame(self, mocker):
        """A geojson with every feature filtered out must degrade to an
        empty areas frame, matching the tabular loader's behaviour."""
        mocker.patch(
            "asf_heat_pump_suitability.getters.base_getters.load_gdf_from_s3_geojson",
            side_effect=ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column "
                "is not supported"
            ),
        )
        df = compare_versions.load_df_cluster_areas("s3://bucket/dir/output.geojson")
        assert df.is_empty(), "a zero-feature geojson must degrade to an empty frame"
        assert df.columns == [
            "area_m2"
        ], "the empty frame must still carry the area column for downstream sums"

    def test_unreadable_file_type_raises(self):
        """File types the comparison cannot read geometry from fail loudly."""
        with pytest.raises(ValueError, match="csv"):
            compare_versions.load_df_cluster_areas("s3://bucket/dir/output.csv")

    def test_geojson_layer_column_is_carried_alongside_areas(self, mocker):
        """A multi-layer front-end geojson's areas must keep each feature's
        layer, so the checks can filter to clusters and the per-layer table
        can tally the rest."""
        gdf = self.gdf_two_squares("EPSG:4326")
        gdf["layer"] = [compare_versions.CLUSTER_LAYER, "ward_boundaries"]
        mocker.patch(
            "asf_heat_pump_suitability.getters.base_getters.load_gdf_from_s3_geojson",
            return_value=gdf,
        )
        df = compare_versions.load_df_cluster_areas("s3://bucket/dir/output.geojson")
        assert df.columns == [
            "area_m2",
            "layer",
        ], "a layered source must yield areas alongside their layer"
        assert df["layer"].to_list() == [
            compare_versions.CLUSTER_LAYER,
            "ward_boundaries",
        ], "each area row must keep its feature's layer"

    def test_geoparquet_layer_column_is_carried_alongside_areas(self, tmp_path):
        """The loader carries `layer` from geoparquet too, so a future
        layered parquet output filters the same way as the geojson."""
        path = tmp_path / "clusters.parquet"
        gdf = self.gdf_two_squares("EPSG:27700")
        gdf["layer"] = [compare_versions.CLUSTER_LAYER, "anchor_loads"]
        gdf.to_parquet(path)
        df = compare_versions.load_df_cluster_areas(str(path))
        assert df["layer"].to_list() == [
            compare_versions.CLUSTER_LAYER,
            "anchor_loads",
        ], "geoparquet area rows must keep their feature's layer too"


class TestFilterDfClustersLayer:
    """Tests for `filter_df_clusters_layer`."""

    def test_keeps_only_the_configured_clusters_layer(self):
        """A multi-layer front-end output bundles whole-county ward polygons
        with the clusters; the filter must keep clusters-layer rows only so
        the wards cannot swamp the cluster checks."""
        df = pl.DataFrame(
            {
                "cluster_id": ["HP_1", None, "HP_2"],
                "layer": [
                    compare_versions.CLUSTER_LAYER,
                    "ward_boundaries",
                    compare_versions.CLUSTER_LAYER,
                ],
            }
        )
        filtered = compare_versions.filter_df_clusters_layer(df)
        assert filtered.height == 2, "only the two clusters-layer rows may survive"
        assert filtered["layer"].unique().to_list() == [
            compare_versions.CLUSTER_LAYER
        ], "no other layer's rows may survive the filter"

    def test_frame_without_a_layer_column_passes_through_unchanged(
        self, df_clusters_old
    ):
        """Pre-layers outputs have no `layer` column and are treated as
        all-clusters, so old-vs-new comparisons across the format change
        keep working."""
        assert (
            compare_versions.filter_df_clusters_layer(df_clusters_old)
            is df_clusters_old
        ), "a frame without a layer column must pass through as the same frame"

    def test_clusters_layer_name_comes_from_config(self):
        """The clusters layer name is config, not code, and must match the
        front-end file's real layer value (verified against the 20260806
        East Lothian output)."""
        assert (
            config["compare_versions"]["cluster_layer"]
            == "clusters_with_contextual_features"
        ), "base.yaml must name the front-end file's real clusters layer"
        assert (
            compare_versions.CLUSTER_LAYER
            == config["compare_versions"]["cluster_layer"]
        ), "the module must read the clusters layer name from config"


class TestGenerateDictDistributionStats:
    """Tests for `generate_dict_distribution_stats`."""

    def test_reports_quartiles_min_max_and_mean(self):
        """The shared helper reports Q1 and Q3 (both quartiles, so a shift
        and a widening stay distinguishable) plus min, max and mean."""
        df = pl.DataFrame({"n_UPRNs": [1, 2, 3, 4, 5]})
        stats = compare_versions.generate_dict_distribution_stats(df, "n_UPRNs")
        assert stats == {
            "min": 1,
            "q1": 2.0,
            "mean": 3.0,
            "q3": 4.0,
            "max": 5,
        }, "the helper must report Q1, Q3, min, max and mean for the column"

    def test_nulls_are_excluded_from_the_statistics(self):
        """Null values must not drag the statistics; they are dropped."""
        df = pl.DataFrame({"n_UPRNs": [1, 2, 3, 4, 5, None]})
        stats = compare_versions.generate_dict_distribution_stats(df, "n_UPRNs")
        assert stats["mean"] == 3.0, "nulls must be excluded, not counted as zeros"

    def test_missing_column_returns_none(self, df_clusters_old):
        """A version without the target column cannot be summarised."""
        assert (
            compare_versions.generate_dict_distribution_stats(
                df_clusters_old, "n_UPRNs"
            )
            is None
        ), "a missing column must degrade to None, not raise"

    def test_all_null_column_returns_none(self):
        """A column with no values has no distribution to report."""
        df = pl.DataFrame({"n_UPRNs": [None, None]}, schema={"n_UPRNs": pl.Int64})
        assert (
            compare_versions.generate_dict_distribution_stats(df, "n_UPRNs") is None
        ), "an all-null column must degrade to None, not report null statistics"


class TestDistributionColumns:
    """Tests for the configured `DISTRIBUTION_COLUMNS` per-stage lists."""

    def test_every_configured_stage_is_a_known_stage(self):
        """Distribution columns are keyed by real pipeline stages."""
        assert set(compare_versions.DISTRIBUTION_COLUMNS) <= set(
            compare_versions.STAGE_OUTPUT_DATASETS
        ), "distribution columns must be configured under known stage names"

    def test_uprns_per_cluster_is_configured_at_the_contextual_stage(self):
        """n_UPRNs is computed at the contextual-features stage, so its
        distribution is configured there — config lists real columns only."""
        assert (
            "n_UPRNs"
            in compare_versions.DISTRIBUTION_COLUMNS["compute_contextual_features"]
        ), "the UPRNs-per-cluster distribution must target the real n_UPRNs column"


class TestGetDictDistributionFrames:
    """Tests for `get_dict_distribution_frames`."""

    def test_cluster_stage_gets_the_derived_area_distribution_only(
        self, df_clusters_old, df_areas_old, df_areas_merged
    ):
        """The cluster stage output has no configured distribution columns;
        its only distribution is the geometry-derived area."""
        frames = compare_versions.get_dict_distribution_frames(
            "cluster",
            df_clusters_old,
            df_clusters_old,
            df_areas_old,
            df_areas_merged,
        )
        assert list(frames) == [
            "area_m2"
        ], "the cluster stage must get exactly the derived area distribution"
        assert frames["area_m2"] == (
            df_areas_old,
            df_areas_merged,
        ), "the area distribution must read the geometry-derived frames"

    def test_contextual_stage_adds_the_configured_columns(
        self, df_clusters_old, df_areas_old
    ):
        """The contextual-features stage gets the derived area plus its
        configured columns (n_UPRNs), read from the tabular output."""
        frames = compare_versions.get_dict_distribution_frames(
            "compute_contextual_features",
            df_clusters_old,
            df_clusters_old,
            df_areas_old,
            df_areas_old,
        )
        assert set(frames) == {
            "area_m2",
            "n_UPRNs",
        }, "the contextual stage must get the area and n_UPRNs distributions"
        assert frames["n_UPRNs"] == (
            df_clusters_old,
            df_clusters_old,
        ), "configured columns must read the tabular stage output"

    def test_stage_without_geometry_gets_no_distributions(self, df_old):
        """Non-geometry stages have no areas and no configured columns yet,
        so no distribution sections are added."""
        frames = compare_versions.get_dict_distribution_frames(
            "decision_tree", df_old, df_old, None, None
        )
        assert frames == {}, "a stage without geometry or configured columns gets none"

    def test_layered_frames_are_filtered_to_the_clusters_layer(self):
        """Multi-layer outputs bundle ward polygons and anchor loads with
        the clusters; every distribution (and so every plot fed from here)
        must cover the clusters layer only."""
        df_tabular = pl.DataFrame(
            {
                "n_UPRNs": [5, 8, None],
                "layer": [
                    compare_versions.CLUSTER_LAYER,
                    compare_versions.CLUSTER_LAYER,
                    "ward_boundaries",
                ],
            }
        )
        df_areas = pl.DataFrame(
            {
                "area_m2": [100.0, 200.0, 5_000_000.0],
                "layer": [
                    compare_versions.CLUSTER_LAYER,
                    compare_versions.CLUSTER_LAYER,
                    "ward_boundaries",
                ],
            }
        )
        frames = compare_versions.get_dict_distribution_frames(
            "compute_contextual_features",
            df_tabular,
            df_tabular,
            df_areas,
            df_areas,
        )
        assert frames["n_UPRNs"][0]["n_UPRNs"].drop_nulls().to_list() == [
            5,
            8,
        ], "ward rows must not enter the n_UPRNs distribution"
        assert frames["area_m2"][1]["area_m2"].to_list() == [
            100.0,
            200.0,
        ], "whole-county ward polygons must not enter the area distribution"


class TestPlotDistributionOverlay:
    """Tests for `plot_distribution_overlay`."""

    def test_saves_a_png_at_the_given_path(self, tmp_path):
        """The overlaid old-vs-new histogram is saved as a PNG file."""
        path = tmp_path / "cluster_plymouth_area_m2.png"
        compare_versions.plot_distribution_overlay(
            pl.Series("area_m2", [100.0, 100.0, 100.0]),
            pl.Series("area_m2", [210.0, 100.0]),
            "area_m2",
            path,
        )
        assert path.exists(), "the plot must be saved at the given path"
        assert path.stat().st_size > 0, "the saved plot must not be an empty file"


class TestGenerateDictDistributionPlots:
    """Tests for `generate_dict_distribution_plots`."""

    def test_saves_one_plot_per_distribution(
        self, tmp_path, df_areas_old, df_areas_merged, df_clusters_old
    ):
        """Each distribution with values on both sides gets one PNG, named
        after the report stem, and the mapping points the report at it."""
        df_uprns = df_clusters_old.with_columns(pl.lit(5).alias("n_UPRNs"))
        frames = {
            "area_m2": (df_areas_old, df_areas_merged),
            "n_UPRNs": (df_uprns, df_uprns),
        }
        plot_files = compare_versions.generate_dict_distribution_plots(
            frames, tmp_path, "cluster_plymouth_20260601_vs_20260722"
        )
        assert plot_files == {
            "area_m2": "cluster_plymouth_20260601_vs_20260722_area_m2.png",
            "n_UPRNs": "cluster_plymouth_20260601_vs_20260722_n_UPRNs.png",
        }, "each distribution must map to its stem-named PNG"
        for filename in plot_files.values():
            assert (tmp_path / filename).exists(), "every mapped PNG must be saved"

    def test_skips_a_distribution_with_no_values_on_one_side(
        self, tmp_path, df_areas_old
    ):
        """A distribution empty on one side (e.g. zero-feature geojson) has
        nothing to overlay: no file, no mapping entry, no crash."""
        empty = pl.DataFrame(schema={"area_m2": pl.Float64})
        plot_files = compare_versions.generate_dict_distribution_plots(
            {"area_m2": (df_areas_old, empty)}, tmp_path, "stem"
        )
        assert plot_files == {}, "an empty side must skip the plot, not crash"
        assert list(tmp_path.iterdir()) == [], "no file may be written for a skip"

    def test_skips_a_distribution_whose_column_is_missing(
        self, tmp_path, df_clusters_old
    ):
        """A configured column missing from one version (schema change) is
        skipped; the report's stats section already notes the gap."""
        with_col = df_clusters_old.with_columns(pl.lit(5).alias("n_UPRNs"))
        plot_files = compare_versions.generate_dict_distribution_plots(
            {"n_UPRNs": (with_col, df_clusters_old)}, tmp_path, "stem"
        )
        assert plot_files == {}, "a missing column must skip the plot, not raise"


class TestParseArguments:
    """Tests for `parse_arguments`."""

    def test_single_release_date_is_a_cli_error_naming_both_options(
        self, mocker, capsys
    ):
        """Passing exactly one date is ambiguous: the CLI must error naming
        both date options rather than silently defaulting the other."""
        mocker.patch(
            "sys.argv",
            [
                "compare_versions.py",
                "--stage",
                "decision_tree",
                "--local_authority",
                "plymouth",
                "--old_release_date",
                "20260601",
            ],
        )
        with pytest.raises(SystemExit):
            compare_versions.parse_arguments()
        stderr = capsys.readouterr().err
        assert (
            "--old_release_date" in stderr and "--new_release_date" in stderr
        ), "the error message must name both date options"

    def test_dates_and_trigger_are_optional_and_default_to_none(self, mocker):
        """Omitting both dates and the trigger is the supported
        compare-latest-two, raw-numbers-only invocation."""
        mocker.patch(
            "sys.argv",
            [
                "compare_versions.py",
                "--stage",
                "decision_tree",
                "--local_authority",
                "plymouth",
            ],
        )
        args = compare_versions.parse_arguments()
        assert (
            args.old_release_date is None
            and args.new_release_date is None
            and args.trigger is None
        ), "omitted dates and trigger must parse as None, not error or default"
