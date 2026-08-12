"""Tests for the S3 input preflight check in `pipeline.validate.check_inputs`."""

import pytest

from asf_heat_pump_suitability.pipeline.validate.check_inputs import (
    generate_list_expanded_square_paths,
    get_list_missing_s3_paths,
    get_list_s3_paths,
    get_str_common_prefix,
)


class FakeS3Client:
    """Minimal stand-in for a boto3 S3 client's `list_objects_v2` method."""

    def __init__(self, existing_paths: list):
        """Store full s3:// paths of the objects that "exist" in the fake S3."""
        self.existing_paths = existing_paths

    def list_objects_v2(self, **kwargs) -> dict:
        """Return KeyCount 1 if any existing path sits under Bucket/Prefix, else 0.

        KeyCount is only ever 0 or 1, mirroring the real client's response to
        a MaxKeys=1 request.
        """
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
        ], "every nested s3:// leaf should be collected, at any depth"

    def test_ignores_non_s3_leaves(self):
        """Leaves that are not s3:// strings (numbers, plain strings, URLs) are skipped."""
        config_section = {
            "threshold": 261.6,
            "name": "not a path",
            "url": "https://github.com/nestauk",
            "path": "s3://bucket/file.csv",
        }
        assert get_list_s3_paths(config_section) == ["s3://bucket/file.csv"], (
            "only strings starting with s3:// should be collected; numbers, "
            "plain strings and non-S3 URLs should be skipped"
        )


class TestGenerateListExpandedSquarePaths:
    """Tests for `generate_list_expanded_square_paths`."""

    def test_square_token_expanded_per_square(self):
        """A {square}-templated path becomes one path per grid square."""
        paths = ["s3://bucket/data/{square}/{square}_{layer}.shp"]
        assert generate_list_expanded_square_paths(paths, ["SX", "SD"]) == [
            "s3://bucket/data/SX/SX_{layer}.shp",
            "s3://bucket/data/SD/SD_{layer}.shp",
        ], "every {square} occurrence should be substituted once per grid square"

    def test_path_without_square_token_unchanged(self):
        """Paths without a {square} token pass through unchanged."""
        paths = ["s3://bucket/inputs/uprn.zip"]
        assert (
            generate_list_expanded_square_paths(paths, ["SX", "SD"]) == paths
        ), "paths without a {square} token should pass through unchanged"


class TestGetStrCommonPrefix:
    """Tests for `get_str_common_prefix`."""

    def test_truncates_templated_path_at_first_brace(self):
        """A templated path is truncated to the prefix before the first '{'."""
        path = "s3://bucket/inputs/opmplc/data/{square}/{square}_{layer}.shp"
        assert (
            get_str_common_prefix(path) == "s3://bucket/inputs/opmplc/data/"
        ), "templated path should truncate to the prefix before the first '{'"

    def test_plain_path_is_unchanged(self):
        """A path without template tokens is returned as-is."""
        path = "s3://bucket/inputs/uprn.zip"
        assert (
            get_str_common_prefix(path) == path
        ), "path without template tokens should be returned unchanged"

    def test_mid_segment_token_truncates_within_segment(self):
        """A template token mid-segment truncates within that segment, not at a '/'."""
        path = (
            "s3://bucket/inputs/opmplc/data/grid_square_{square}/{square}_{layer}.shp"
        )
        assert (
            get_str_common_prefix(path) == "s3://bucket/inputs/opmplc/data/grid_square_"
        ), "prefix should cut at the first '{' even when the token is mid-segment"


class TestGetListMissingS3Paths:
    """Tests for `get_list_missing_s3_paths`."""

    @pytest.fixture(scope="class")
    @classmethod
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
        assert (
            get_list_missing_s3_paths(paths, s3_client) == []
        ), "no paths should be reported missing when every configured path exists"

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
        ], "every missing path should be reported in one pass, not just the first"

    def test_missing_square_reported_individually(self, s3_client):
        """After expansion, a square with no files is reported while present squares pass."""
        paths = generate_list_expanded_square_paths(
            ["s3://bucket/inputs/opmplc/data/{square}/{square}_{layer}.shp"],
            ["SX", "SD"],
        )
        assert get_list_missing_s3_paths(paths, s3_client) == [
            "s3://bucket/inputs/opmplc/data/SD/SD_{layer}.shp"
        ], "the empty SD square should be reported missing while SX passes"
