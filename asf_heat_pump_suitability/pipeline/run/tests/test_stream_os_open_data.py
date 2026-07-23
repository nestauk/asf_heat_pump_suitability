"""
Tests for the pure helpers in pipeline/run/stream_os_open_data.py.

Uses a trimmed copy of real OS Downloads API responses
(os_downloads_api_fixture.json) so no network access is needed. The fixture
keeps every download format the API offers for areas HW, HT and GB, so the
tests pin down both the shapefile-format filter and the per-area vs GB split.
"""

import json
from pathlib import Path

import pytest

from asf_heat_pump_suitability.pipeline.run import stream_os_open_data

FIXTURE_PATH = Path(__file__).parent / "os_downloads_api_fixture.json"


@pytest.fixture(scope="module")
def api_fixture() -> dict:
    """Trimmed OS Downloads API responses for the three products."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


class TestFilterListShapefileDownloads:
    """Tests for `filter_list_shapefile_downloads`."""

    def test_per_area_product_keeps_only_per_area_shapefiles(self, api_fixture):
        """OpenMapLocal keeps per-area ESRI Shapefile zips, dropping the GB zip and other formats."""
        downloads = api_fixture["OpenMapLocal"]["downloads"]
        result = stream_os_open_data.filter_list_shapefile_downloads(
            downloads, "OpenMapLocal"
        )
        assert sorted(d["area"] for d in result) == ["HT", "HW"]
        assert all(d["format"] == "ESRI® Shapefile" for d in result)

    def test_gb_zip_product_keeps_only_gb_shapefile(self, api_fixture):
        """OpenRoads keeps the single GB ESRI Shapefile zip, dropping GML/GeoPackage/Vector Tiles."""
        downloads = api_fixture["OpenRoads"]["downloads"]
        result = stream_os_open_data.filter_list_shapefile_downloads(
            downloads, "OpenRoads"
        )
        assert len(result) == 1
        assert result[0]["area"] == "GB"
        assert result[0]["fileName"] == "oproad_essh_gb.zip"

    def test_greenspace_drops_gb_entries(self, api_fixture):
        """OpenGreenspace keeps per-area shapefile zips only, excluding the GB shapefile zip."""
        downloads = api_fixture["OpenGreenspace"]["downloads"]
        result = stream_os_open_data.filter_list_shapefile_downloads(
            downloads, "OpenGreenspace"
        )
        assert [d["area"] for d in result] == ["HT"]
        assert result[0]["fileName"] == "opgrsp_essh_ht.zip"


class TestGenerateKeyZipMember:
    """Tests for `generate_key_zip_member`."""

    def test_openmaplocal_data_member_nested_under_square_dir(self):
        """OpenMapLocal per-area zip data members map to data/{square}/{square}_{layer}.{ext}."""
        key = stream_os_open_data.generate_key_zip_member(
            "OpenMapLocal",
            "HW",
            "OS OpenMap Local (ESRI Shape File) HW/data/HW_Building.shp",
        )
        assert key == "data/HW/HW_Building.shp"

    def test_openmaplocal_sidecars_follow_data_member(self):
        """All shapefile sidecar extensions map alongside the .shp file."""
        for ext in ("dbf", "prj", "shx", "cpg"):
            key = stream_os_open_data.generate_key_zip_member(
                "OpenMapLocal",
                "SX",
                f"OS OpenMap Local (ESRI Shape File) SX/data/SX_Building.{ext}",
            )
            assert key == f"data/SX/SX_Building.{ext}"

    def test_openmaplocal_doc_and_readme_at_prefix_root(self):
        """OpenMapLocal licence/readme land once at the prefix root, matching the current layout."""
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenMapLocal",
                "HW",
                "OS OpenMap Local (ESRI Shape File) HW/doc/licence.txt",
            )
            == "doc/licence.txt"
        )
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenMapLocal",
                "HW",
                "OS OpenMap Local (ESRI Shape File) HW/readme.txt",
            )
            == "readme.txt"
        )

    def test_openroads_members_map_unchanged(self):
        """OpenRoads GB zip members keep their in-zip paths (flat data/ layout)."""
        key = stream_os_open_data.generate_key_zip_member(
            "OpenRoads", "GB", "data/HP_RoadLink.shp"
        )
        assert key == "data/HP_RoadLink.shp"

    def test_openroads_leading_slash_members_stripped(self):
        """The OpenRoads GB zip names doc/readme members with a leading slash; it is stripped."""
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenRoads", "GB", "/doc/licence.txt"
            )
            == "doc/licence.txt"
        )
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenRoads", "GB", "/readme.txt"
            )
            == "readme.txt"
        )

    def test_greenspace_members_nested_under_square(self):
        """OpenGreenspace members map to {square}/data|doc|readme, matching the current layout."""
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenGreenspace",
                "HT",
                "OS Open Greenspace (ESRI Shape File) HT/data/HT_AccessPoint.shp",
            )
            == "HT/data/HT_AccessPoint.shp"
        )
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenGreenspace",
                "HT",
                "OS Open Greenspace (ESRI Shape File) HT/doc/licence.txt",
            )
            == "HT/doc/licence.txt"
        )

    def test_directory_entries_skipped(self):
        """Zip directory entries map to None for every product."""
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenMapLocal", "HW", "OS OpenMap Local (ESRI Shape File) HW/data/"
            )
            is None
        )
        assert (
            stream_os_open_data.generate_key_zip_member("OpenRoads", "GB", "/data/")
            is None
        )
        assert (
            stream_os_open_data.generate_key_zip_member(
                "OpenGreenspace", "HT", "OS Open Greenspace (ESRI Shape File) HT/"
            )
            is None
        )

    def test_unknown_product_raises(self):
        """A product with no layout mapping fails loudly."""
        with pytest.raises(ValueError, match="OpenRivers"):
            stream_os_open_data.generate_key_zip_member(
                "OpenRivers", "HW", "data/HW_River.shp"
            )


class TestGenerateDictReconciliation:
    """Tests for `generate_dict_reconciliation`."""

    def test_matching_sets_give_empty_diff(self):
        """Identical expected and actual key sets reconcile cleanly."""
        keys = {"data/HP_RoadLink.shp", "readme.txt"}
        diff = stream_os_open_data.generate_dict_reconciliation(keys, keys)
        assert diff == {"missing": [], "unexpected": []}

    def test_missing_and_unexpected_keys_reported_sorted(self):
        """Keys absent from S3 are 'missing'; extra S3 keys are 'unexpected'; both sorted."""
        expected = {"data/HZ_RoadLink.shp", "data/HP_RoadLink.shp"}
        actual = {"data/HP_RoadLink.shp", ".DS_Store"}
        diff = stream_os_open_data.generate_dict_reconciliation(expected, actual)
        assert diff == {
            "missing": ["data/HZ_RoadLink.shp"],
            "unexpected": [".DS_Store"],
        }
