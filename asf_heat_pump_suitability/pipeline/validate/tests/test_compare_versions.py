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
        assert counts["rows_delta"] == 0
        assert counts["uprns_delta"] == 0

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
        }

    def test_uprn_counts_are_none_without_uprn_column(self, df_old):
        """Outputs without a UPRN column (e.g. clusters) get row counts only."""
        no_uprn = df_old.drop("UPRN")
        counts = compare_versions.generate_dict_count_delta(no_uprn, no_uprn)
        assert counts["rows_delta"] == 0
        assert counts["uprns_old"] is None
        assert counts["uprns_new"] is None
        assert counts["uprns_delta"] is None


class TestGenerateDictSchemaDiff:
    """Tests for `generate_dict_schema_diff`."""

    def test_identical_schemas_have_empty_diff(self, df_old, df_new_identical):
        """No drift means no added, removed or retyped columns."""
        assert compare_versions.generate_dict_schema_diff(df_old, df_new_identical) == {
            "added": {},
            "removed": {},
            "dtype_changed": {},
        }

    def test_reports_added_and_removed_columns_with_dtypes(self, df_old):
        """Column additions and removals are reported with their dtypes."""
        df_new = df_old.drop("in_hn_zone").with_columns(
            pl.lit(1.5).alias("garden_area_m2")
        )
        diff = compare_versions.generate_dict_schema_diff(df_old, df_new)
        assert diff["added"] == {"garden_area_m2": "Float64"}
        assert diff["removed"] == {"in_hn_zone": "Boolean"}

    def test_reports_dtype_changes_as_old_new_pairs(self, df_old):
        """A column changing dtype between versions is reported old -> new."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Utf8))
        diff = compare_versions.generate_dict_schema_diff(df_old, df_new)
        assert diff["dtype_changed"] == {"UPRN": ("Int64", "String")}


class TestGenerateDictUprnChurn:
    """Tests for `generate_dict_uprn_churn`."""

    def test_identical_versions_have_no_churn(self, df_old, df_new_identical):
        """No drift means every UPRN is retained."""
        assert compare_versions.generate_dict_uprn_churn(df_old, df_new_identical) == {
            "n_added": 0,
            "n_removed": 0,
            "n_retained": 4,
            "removed_share": 0.0,
        }

    def test_counts_added_removed_and_retained_uprns(self, df_old, df_new_churned):
        """Added/removed/retained are set differences on the UPRN key, and
        removed_share is the fraction of old UPRNs lost."""
        assert compare_versions.generate_dict_uprn_churn(df_old, df_new_churned) == {
            "n_added": 2,
            "n_removed": 1,
            "n_retained": 3,
            "removed_share": 0.25,
        }

    def test_returns_none_without_uprn_column(self, df_old):
        """Outputs without a UPRN column cannot be churn-checked."""
        no_uprn = df_old.drop("UPRN")
        assert compare_versions.generate_dict_uprn_churn(no_uprn, no_uprn) is None

    def test_matches_uprns_across_dtype_changes(self, df_old):
        """A UPRN dtype change between versions must not read as full churn."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Utf8))
        churn = compare_versions.generate_dict_uprn_churn(df_old, df_new)
        assert churn["n_retained"] == 4
        assert churn["n_removed"] == 0

    def test_matches_uprns_across_int_and_float_dtypes(self, df_old):
        """An Int64-vs-Float64 UPRN mismatch (e.g. a pandas null-upcast) must
        not read as full churn: casting straight to Utf8 would compare "123"
        against "123.0" and miss every match."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Float64))
        churn = compare_versions.generate_dict_uprn_churn(df_old, df_new)
        assert churn["n_retained"] == 4
        assert churn["n_removed"] == 0


class TestGenerateStrChurnNote:
    """Tests for `generate_str_churn_note`."""

    def test_expected_churn_within_tolerance_gets_no_note(self):
        """Churn at or below the rubric's tolerance is expected: no warning."""
        churn = {"n_added": 2, "n_removed": 1, "n_retained": 3, "removed_share": 0.25}
        assert (
            compare_versions.generate_str_churn_note(churn, max_removed_share=0.25)
            is None
        )

    def test_unexpected_uprn_loss_above_tolerance_gets_warning(self):
        """UPRN loss above the rubric's tolerance produces a warning naming
        the observed share and the tolerance."""
        churn = {"n_added": 0, "n_removed": 3, "n_retained": 1, "removed_share": 0.75}
        note = compare_versions.generate_str_churn_note(churn, max_removed_share=0.05)
        assert "75.0%" in note
        assert "5.0%" in note


class TestGetDictTolerances:
    """Tests for `get_dict_tolerances`."""

    @pytest.mark.parametrize("trigger", ["methodology_change", "input_release"])
    def test_each_trigger_rubric_has_a_removed_uprn_tolerance(self, trigger):
        """Both rubrics are configured in base.yaml with the churn tolerance."""
        tolerances = compare_versions.get_dict_tolerances(trigger)
        assert isinstance(tolerances["max_removed_uprn_share"], float)

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
        ).is_empty()
        assert transitions["n_uprns"].sum() == 4

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
        assert moved["n_uprns"].to_list() == [1]
        # Only the 3 retained UPRNs are counted; added/removed ones are churn.
        assert transitions["n_uprns"].sum() == 3

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
        assert nulled["assigned_tech_old"].to_list() == ["Individual solution"]
        assert transitions["assigned_tech_new"].null_count() == 0

    def test_returns_none_without_tech_column(self, df_old, df_new_identical):
        """A version missing `assigned_tech` must degrade to None rather than
        raise `ColumnNotFoundError` — this data function must be as safe to
        call directly as its `generate_dict_*` siblings."""
        df_new = df_new_identical.drop("assigned_tech")
        assert compare_versions.generate_df_tech_transitions(df_old, df_new) is None

    def test_duplicate_uprns_do_not_inflate_transition_counts(self, df_old):
        """A duplicated UPRN in one version must not cross-product into an
        inflated transition count; each version is deduplicated on UPRN
        first."""
        df_new = pl.concat([df_old, df_old.head(1)])  # UPRN 1 appears twice
        transitions = compare_versions.generate_df_tech_transitions(df_old, df_new)
        assert transitions["n_uprns"].sum() == 4

    def test_matches_uprns_across_int_and_float_dtypes(self, df_old):
        """An Int64-vs-Float64 UPRN mismatch must not read as churn here
        either, since the transition matrix joins on the same UPRN key."""
        df_new = df_old.with_columns(pl.col("UPRN").cast(pl.Float64))
        transitions = compare_versions.generate_df_tech_transitions(df_old, df_new)
        assert transitions["n_uprns"].sum() == 4


class TestLoadDictManifest:
    """Tests for `load_dict_manifest`."""

    def test_loads_the_outputs_colocated_manifest(self, mocker):
        """The manifest is read from the output's co-located .manifest.json."""
        manifest = {"stage": "decision_tree", "git_commit": "a" * 40}
        opened = mocker.patch(
            "fsspec.open", mocker.mock_open(read_data=json.dumps(manifest))
        )
        loaded = compare_versions.load_dict_manifest("s3://bucket/dir/output.parquet")
        assert loaded == manifest
        assert opened.call_args.args[0] == "s3://bucket/dir/output.manifest.json"

    def test_missing_manifest_returns_none(self, mocker):
        """Pre-#440 outputs have no manifest: degrade to None, don't raise."""
        mocker.patch("fsspec.open", side_effect=FileNotFoundError("no such key"))
        assert (
            compare_versions.load_dict_manifest("s3://bucket/dir/output.parquet")
            is None
        )


class TestGenerateDictInputVersionChanges:
    """Tests for `generate_dict_input_version_changes`."""

    def test_identical_input_versions_have_empty_changes(self):
        """No input re-release means no changed, added or removed inputs."""
        versions = {"epc.domestic": "s3://bucket/inputs/2026Q1_epc.parquet"}
        assert compare_versions.generate_dict_input_version_changes(
            {"input_versions": versions}, {"input_versions": versions}
        ) == {"changed": {}, "added": {}, "removed": {}}

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
        }


class TestStageModulePaths:
    """Tests for the curated `STAGE_MODULE_PATHS` lists."""

    def test_covers_the_same_stages_as_the_run_manifest(self):
        """Commit-log scoping mirrors the run manifest's curated stage list."""
        assert set(compare_versions.STAGE_MODULE_PATHS) == set(
            manifest_utils.STAGE_INPUT_KEYS
        )

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
        assert commits == ["abc1234 Fix tree", "def5678 Tune zones"]
        command = run.call_args.args[0]
        assert f"{'a' * 40}..{'b' * 40}" in command
        paths = command[command.index("--") + 1 :]
        assert paths == compare_versions.STAGE_MODULE_PATHS["decision_tree"]

    def test_same_commit_returns_empty_log_without_running_git(self, mocker):
        """Two outputs from the same commit cannot differ by code: empty log."""
        run = mocker.patch("subprocess.run")
        assert (
            compare_versions.generate_list_commit_log("a" * 40, "a" * 40, "uprns") == []
        )
        run.assert_not_called()

    def test_unknown_recorded_commit_returns_none(self, mocker):
        """A manifest recording the "unknown" commit sentinel cannot be scoped."""
        run = mocker.patch("subprocess.run")
        assert (
            compare_versions.generate_list_commit_log("unknown", "b" * 40, "uprns")
            is None
        )
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
        )
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
        )
        assert run.call_count == 2


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
        )

    def test_fills_the_clustering_tolerance_placeholder(self):
        """The contextual-features template's {tolerance_m} placeholder is
        filled from config so the path matches what the pipeline saved."""
        tolerance_m = config["constant"]["clustering"]["tolerance_m"]
        path = compare_versions.get_str_stage_output_path(
            "compute_contextual_features", "plymouth", "20260722"
        )
        assert path.endswith(f"_clusters_contextual_features_{tolerance_m}m.geojson")

    def test_covers_the_same_stages_as_the_run_manifest(self):
        """Output-dataset mapping mirrors the run manifest's curated stages."""
        assert set(compare_versions.STAGE_OUTPUT_DATASETS) == set(
            manifest_utils.STAGE_INPUT_KEYS
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
        assert "methodology_change" in report
        assert "20260601" in report
        assert "20260722" in report

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
        assert "Row and UPRN counts" in report
        assert "Schema diff" in report
        assert "UPRN churn" in report
        assert "2026Q2_epc" in report
        assert "abc1234 Tune decision tree" in report

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
        assert "Tech-assignment transitions" in decision_tree
        assert "Tech-assignment transitions" not in other

    def test_missing_manifests_name_both_versions_in_the_note(
        self, df_old, df_new_identical
    ):
        """When both versions lack a manifest, the note must not attribute
        the gap to only one of them."""
        report = generate_report(df_old, df_new_identical, None, None)
        assert "manifest missing for the old and new versions" in report.lower()
        assert "Row and UPRN counts" in report

    def test_missing_manifest_names_only_the_version_that_lacks_one(
        self, df_old, df_new_identical, manifests
    ):
        """When only one version lacks a manifest, the note names that
        version specifically, not both."""
        _, manifest_new = manifests
        report = generate_report(df_old, df_new_identical, None, manifest_new)
        assert "manifest missing for the old version " in report.lower()
        assert "old and new" not in report.lower()

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
        assert "Skipped" in report
        assert "assigned_tech" in report

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
        assert "No UPRNs retained across versions; matrix skipped." in report

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
        assert "assigned_tech_old" in report

    def test_unexpected_uprn_loss_warning_appears(self, df_old, manifests, mocker):
        """UPRN loss above the rubric tolerance surfaces as a warning line."""
        mocker.patch.object(
            compare_versions, "generate_list_commit_log", return_value=[]
        )
        df_new = df_old.head(1)
        report = generate_report(df_old, df_new, *manifests)
        assert "WARNING" in report


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
        assert "geometry" not in df.columns
        assert df["UPRN"].to_list() == [1, 2]

    def test_loads_plain_parquet_unchanged(self, tmp_path):
        """Polars-written outputs have no geometry and load as-is."""
        path = tmp_path / "output.parquet"
        pl.DataFrame({"UPRN": [1, 2], "in_hn_zone": [True, False]}).write_parquet(path)
        df = compare_versions.load_transform_df_stage_output(str(path))
        assert df.columns == ["UPRN", "in_hn_zone"]
        assert df.height == 2

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
        assert df.is_empty()
