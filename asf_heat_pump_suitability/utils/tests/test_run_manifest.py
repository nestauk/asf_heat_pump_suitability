"""
Tests for asf_heat_pump_suitability.utils.run_manifest.

Run:
pytest asf_heat_pump_suitability/utils/tests/test_run_manifest.py
"""

import json
import re
from datetime import datetime

import pytest

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import run_manifest


class TestGetStrGitCommit:
    """Tests for `get_str_git_commit`."""

    def test_returns_current_commit_hash(self):
        """Returns the 40-character hex hash of the repo's HEAD commit."""
        assert re.fullmatch(r"[0-9a-f]{40}", run_manifest.get_str_git_commit())

    def test_returns_unknown_sentinel_when_git_unavailable(self, monkeypatch):
        """Falls back to "unknown" rather than raising when git cannot run."""

        def raise_oserror(*args, **kwargs):
            raise OSError("git not found")

        monkeypatch.setattr(run_manifest.subprocess, "run", raise_oserror)
        assert run_manifest.get_str_git_commit() == "unknown"


class TestGenerateDictInputVersions:
    """Tests for `generate_dict_input_versions`."""

    def test_flattens_nested_config_to_dot_separated_keys(self):
        """Nested dataset path config flattens to one dot-separated key per path."""
        nested = {
            "epc": {"domestic": "s3://a", "commercial": {"EW": "s3://b"}},
            "top_level": "s3://c",
        }
        assert run_manifest.generate_dict_input_versions(nested) == {
            "epc.domestic": "s3://a",
            "epc.commercial.EW": "s3://b",
            "top_level": "s3://c",
        }

    def test_defaults_to_config_data_path_strings(self):
        """With no argument, returns the resolved path strings from config["data"]."""
        input_versions = run_manifest.generate_dict_input_versions()
        assert input_versions["epc.domestic"] == config["data"]["epc"]["domestic"]
        assert all(isinstance(path, str) for path in input_versions.values())


@pytest.fixture(scope="module")
def manifest():
    """Run manifest built once with hand-crafted entrypoint arguments."""
    return run_manifest.generate_dict_run_manifest(
        stage="uprns",
        local_authority="plymouth",
        row_count=123,
        params={"local_authorities": ["plymouth"], "release_date": "20260722"},
    )


class TestGenerateDictRunManifest:
    """Tests for `generate_dict_run_manifest`."""

    def test_contains_exactly_the_expected_keys(self, manifest):
        """Manifest has the seven keys pinned by the spec and no others."""
        assert set(manifest) == {
            "stage",
            "local_authority",
            "run_at",
            "git_commit",
            "input_versions",
            "row_count",
            "params",
        }

    def test_passes_through_stage_local_authority_row_count_params(self, manifest):
        """Caller-supplied values appear unchanged in the manifest."""
        assert manifest["stage"] == "uprns"
        assert manifest["local_authority"] == "plymouth"
        assert manifest["row_count"] == 123
        assert manifest["params"] == {
            "local_authorities": ["plymouth"],
            "release_date": "20260722",
        }

    def test_run_at_is_parseable_iso_timestamp(self, manifest):
        """run_at records the run time as an ISO 8601 timestamp."""
        assert isinstance(datetime.fromisoformat(manifest["run_at"]), datetime)

    def test_git_commit_is_commit_hash_or_unknown_sentinel(self, manifest):
        """git_commit is a 40-character hex hash, or "unknown" when unavailable."""
        assert (
            re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
            or manifest["git_commit"] == "unknown"
        )

    def test_input_versions_snapshot_config_data(self, manifest):
        """input_versions records the resolved config["data"] path strings."""
        assert manifest["input_versions"] == run_manifest.generate_dict_input_versions(
            config["data"]
        )

    def test_manifest_is_json_serialisable(self, manifest):
        """The manifest can be dumped to JSON without custom encoders."""
        assert isinstance(json.dumps(manifest), str)


class TestGetStrManifestPath:
    """Tests for `get_str_manifest_path`."""

    def test_replaces_parquet_extension(self):
        """A parquet output maps to a co-located {basename}.manifest.json."""
        assert (
            run_manifest.get_str_manifest_path(
                "s3://bucket/outputs/data/plymouth/20260722/plymouth_domestic_uprns.parquet"
            )
            == "s3://bucket/outputs/data/plymouth/20260722/plymouth_domestic_uprns.manifest.json"
        )

    def test_replaces_geojson_extension(self):
        """A geojson output maps to a co-located {basename}.manifest.json."""
        assert (
            run_manifest.get_str_manifest_path(
                "s3://bucket/outputs/data/plymouth/20260722/plymouth_clusters_contextual_features_5m.geojson"
            )
            == "s3://bucket/outputs/data/plymouth/20260722/plymouth_clusters_contextual_features_5m.manifest.json"
        )

    def test_filename_never_collides_with_front_end_manifest_json(self):
        """The derived filename keeps the output basename, so it can never be the
        bare `manifest.json` owned by pipeline/run/create_manifest.py."""
        path = run_manifest.get_str_manifest_path("s3://bucket/dir/output.parquet")
        assert path.split("/")[-1] != "manifest.json"
        assert path.endswith(".manifest.json")
