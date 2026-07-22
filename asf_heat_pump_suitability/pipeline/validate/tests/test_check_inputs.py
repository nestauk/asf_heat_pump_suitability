import pytest

from asf_heat_pump_suitability.pipeline.validate.check_inputs import (
    get_list_missing_s3_paths,
    get_list_s3_paths,
    get_str_checkable_prefix,
)


class FakeS3Client:
    """Minimal stand-in for a boto3 S3 client's `list_objects_v2` method."""

    def __init__(self, existing_paths: list):
        """Store full s3:// paths of the objects that "exist" in the fake S3."""
        self.existing_paths = existing_paths

    def list_objects_v2(self, **kwargs) -> dict:
        """Return KeyCount 1 if any existing path sits under Bucket/Prefix."""
        prefix = f"s3://{kwargs['Bucket']}/{kwargs['Prefix']}"
        key_count = int(any(path.startswith(prefix) for path in self.existing_paths))
        return {"KeyCount": key_count}


class TestGetListS3Paths:
    """Tests for `get_list_s3_paths`."""

    def test_collects_leaves_from_nested_mapping(self):
        """All s3:// string leaves are collected, however deeply nested."""
        config_section = {
            "geodata": {
                "uprn": "s3://bucket/inputs/uprn.zip",
                "boundaries": {
                    "lad": "s3://bucket/inputs/boundaries/lad.shp",
                },
            },
            "epc": "s3://other-bucket/lakehouse/epc.parquet",
        }
        assert sorted(get_list_s3_paths(config_section)) == [
            "s3://bucket/inputs/boundaries/lad.shp",
            "s3://bucket/inputs/uprn.zip",
            "s3://other-bucket/lakehouse/epc.parquet",
        ]

    def test_ignores_non_s3_leaves(self):
        """Leaves that are not s3:// strings (numbers, plain strings) are skipped."""
        config_section = {
            "threshold": 261.6,
            "name": "not a path",
            "path": "s3://bucket/file.csv",
        }
        assert get_list_s3_paths(config_section) == ["s3://bucket/file.csv"]


class TestGetStrCheckablePrefix:
    """Tests for `get_str_checkable_prefix`."""

    def test_truncates_templated_path_at_first_brace(self):
        """A templated path is truncated to the prefix before the first '{'."""
        path = "s3://bucket/inputs/opmplc/data/{square}/{square}_{layer}.shp"
        assert get_str_checkable_prefix(path) == "s3://bucket/inputs/opmplc/data/"

    def test_plain_path_is_unchanged(self):
        """A path without template tokens is returned as-is."""
        path = "s3://bucket/inputs/uprn.zip"
        assert get_str_checkable_prefix(path) == path


class TestGetListMissingS3Paths:
    """Tests for `get_list_missing_s3_paths`."""

    @pytest.fixture(scope="class")
    def s3_client(self) -> FakeS3Client:
        """Fake S3 with two objects, one under a templated dataset's prefix."""
        return FakeS3Client(
            existing_paths=[
                "s3://bucket/inputs/uprn.zip",
                "s3://bucket/inputs/opmplc/data/SX/SX_building.shp",
            ]
        )

    def test_all_present_returns_empty_list(self, s3_client):
        """No paths are reported when every configured path exists."""
        paths = [
            "s3://bucket/inputs/uprn.zip",
            "s3://bucket/inputs/opmplc/data/{square}/{square}_{layer}.shp",
        ]
        assert get_list_missing_s3_paths(paths, s3_client) == []

    def test_missing_paths_all_reported_in_one_pass(self, s3_client):
        """Every missing configured path is reported, not just the first."""
        paths = [
            "s3://bucket/inputs/uprn.zip",
            "s3://bucket/inputs/does_not_exist.csv",
            "s3://bucket/inputs/old_prefix/{square}/{square}_{layer}.shp",
        ]
        assert get_list_missing_s3_paths(paths, s3_client) == [
            "s3://bucket/inputs/does_not_exist.csv",
            "s3://bucket/inputs/old_prefix/{square}/{square}_{layer}.shp",
        ]
