"""
Tests for grid-square getters in getters/load_geodata.py.

Missing grid-square files on S3 (e.g. sea-only squares with no road layer) raise
pyogrio DataSourceError and must be skipped, while any other read failure must
still raise.
"""

import geopandas as gpd
import pytest
from pyogrio.errors import DataSourceError
from shapely.geometry import Point

from asf_heat_pump_suitability.getters import load_geodata

SAMPLE_GDF = gpd.GeoDataFrame({"ID": [1], "geometry": [Point(0, 0)]}, crs=27700)


def _fake_read_file_missing_square(missing_square: str):
    """Return a stand-in for gpd.read_file which raises DataSourceError for one square."""

    def fake_read_file(path, **kwargs):
        if missing_square in str(path):
            raise DataSourceError(
                f"'{path}' does not exist in the file system, and is not recognized as a supported dataset name."
            )
        return SAMPLE_GDF.copy()

    return fake_read_file


def test_load_gdf_os_openmap_layer_skips_missing_square(monkeypatch):
    """Squares whose layer file is missing from S3 are skipped; the rest are returned."""
    monkeypatch.setattr(gpd, "read_file", _fake_read_file_missing_square("HX"))
    gdf = load_geodata.load_gdf_os_openmap_layer(
        "important_building", grid_squares=["ND", "HX"]
    )
    assert len(gdf) == 1


def test_load_gdf_os_openroad_skips_missing_square(monkeypatch):
    """Squares whose road file is missing from S3 are skipped; the rest are returned."""
    monkeypatch.setattr(gpd, "read_file", _fake_read_file_missing_square("HX"))
    gdf = load_geodata.load_gdf_os_openroad(grid_squares=["HY", "HX"])
    assert len(gdf) == 1


@pytest.mark.parametrize(
    "getter",
    [
        lambda: load_geodata.load_gdf_os_openmap_layer(
            "important_building", grid_squares=["ND"]
        ),
        lambda: load_geodata.load_gdf_os_openroad(grid_squares=["ND"]),
    ],
)
def test_getters_reraise_other_data_source_errors(monkeypatch, getter):
    """DataSourceErrors unrelated to missing files (e.g. credentials) must not be skipped."""

    def fake_read_file(path, **kwargs):
        raise DataSourceError("AWS error: access denied")

    monkeypatch.setattr(gpd, "read_file", fake_read_file)
    with pytest.raises(DataSourceError):
        getter()
