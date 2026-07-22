"""
Tests for asf_heat_pump_suitability.utils.run_manifest.

Run:
pytest asf_heat_pump_suitability/utils/tests/test_run_manifest.py
"""

import json
import logging
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

    def test_resolves_exactly_the_requested_keys(self):
        """Returns the resolved path string for each requested key and nothing else."""
        assert run_manifest.generate_dict_input_versions(
            ["epc.domestic", "geodata.boundaries.UK_ons_lad_bounds"]
        ) == {
            "epc.domestic": config["data"]["epc"]["domestic"],
            "geodata.boundaries.UK_ons_lad_bounds": config["data"]["geodata"][
                "boundaries"
            ]["UK_ons_lad_bounds"],
        }

    def test_unknown_key_raises_keyerror(self):
        """A key path missing from config["data"] fails loudly at run time rather
        than silently omitting lineage."""
        with pytest.raises(KeyError, match="epc.nonexistent"):
            run_manifest.generate_dict_input_versions(["epc.nonexistent"])

    def test_subtree_key_raises_keyerror(self):
        """A key path resolving to a config subtree rather than a dataset path
        string is a curated-list mistake and fails loudly."""
        with pytest.raises(KeyError, match="subtree"):
            run_manifest.generate_dict_input_versions(["epc"])


class TestStageInputKeys:
    """Tests for the curated `STAGE_INPUT_KEYS` lists."""

    def test_covers_the_five_pipeline_stages(self):
        """One curated list exists per pipeline entrypoint."""
        assert set(run_manifest.STAGE_INPUT_KEYS) == {
            "uprns",
            "add_features",
            "decision_tree",
            "cluster",
            "compute_contextual_features",
        }

    def test_every_curated_key_resolves_to_a_path_string(self):
        """Every curated key resolves in config["data"] to a path string, so a
        config rename cannot silently break a stage's lineage."""
        for stage, input_keys in run_manifest.STAGE_INPUT_KEYS.items():
            input_versions = run_manifest.generate_dict_input_versions(input_keys)
            assert all(
                isinstance(path, str) for path in input_versions.values()
            ), f"Non-string dataset path in curated keys for stage: {stage}"


@pytest.fixture(scope="module")
def manifest():
    """Run manifest built once with hand-crafted entrypoint arguments."""
    return run_manifest.generate_dict_run_manifest(
        stage="uprns",
        local_authority="plymouth",
        row_count=123,
        params={"local_authorities": ["plymouth"], "release_date": "20260722"},
        input_keys=run_manifest.STAGE_INPUT_KEYS["uprns"],
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

    def test_input_versions_resolve_the_curated_stage_keys(self, manifest):
        """input_versions records exactly the curated input keys the caller passed."""
        assert manifest["input_versions"] == run_manifest.generate_dict_input_versions(
            run_manifest.STAGE_INPUT_KEYS["uprns"]
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


class TestSaveManifestToS3:
    """Tests for `save_manifest_to_s3`."""

    def test_write_failure_is_swallowed_and_logged(self, monkeypatch, caplog):
        """A failed manifest write logs a warning naming the manifest path
        instead of raising, so it can never abort a pipeline run."""

        def raise_oserror(*args, **kwargs):
            raise OSError("S3 write failed")

        monkeypatch.setattr(run_manifest.fsspec, "open", raise_oserror)
        with caplog.at_level(logging.WARNING):
            run_manifest.save_manifest_to_s3(
                {"stage": "uprns"}, "s3://bucket/dir/output.parquet"
            )
        assert "s3://bucket/dir/output.manifest.json" in caplog.text
