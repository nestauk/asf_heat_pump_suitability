"""
Tests for asf_heat_pump_suitability.pipeline.validate.compare_versions.

Run:
pytest asf_heat_pump_suitability/pipeline/validate/tests/test_compare_versions.py
"""

import polars as pl
import pytest

from asf_heat_pump_suitability.pipeline.validate import compare_versions


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
