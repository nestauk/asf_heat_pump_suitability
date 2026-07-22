"""
Tests for grid-square getters in getters/load_geodata.py.

Genuine missing grid-square files on S3 (due to true absence of data e.g. sea-only squares
with no road layer) raise pyogrio DataSourceError and must be skipped, while any other read
failure must still raise.
"""

import geopandas as gpd
import pytest
from pyogrio.errors import DataSourceError
from shapely.geometry import Point

from asf_heat_pump_suitability.getters import load_geodata

SAMPLE_GDF = gpd.GeoDataFrame({"ID": [1], "geometry": [Point(0, 0)]}, crs=27700)


def read_file_missing_hx_side_effect(path, **kwargs):
    """Stand-in for gpd.read_file: raise the S3 missing-file DataSourceError for square HX only."""
    if "HX" in str(path):
        raise DataSourceError(
            f"'{path}' does not exist in the file system, and is not recognized as a supported dataset name."
        )
    return SAMPLE_GDF.copy()


class TestLoadGdfOsOpenmapLayer:
    """Tests for `load_gdf_os_openmap_layer`."""

    def test_skips_missing_square(self, mocker):
        """Squares whose layer file is missing from S3 are skipped; the rest are returned."""
        mocker.patch(
            "geopandas.read_file", side_effect=read_file_missing_hx_side_effect
        )
        gdf = load_geodata.load_gdf_os_openmap_layer(
            "important_building", grid_squares=["ND", "HX"]
        )
        assert len(gdf) == 1

    def test_reraises_other_data_source_errors(self, mocker):
        """DataSourceErrors unrelated to missing files (e.g. credentials) must not be skipped."""
        mocker.patch(
            "geopandas.read_file",
            side_effect=DataSourceError("AWS error: access denied"),
        )
        with pytest.raises(DataSourceError):
            load_geodata.load_gdf_os_openmap_layer(
                "important_building", grid_squares=["ND"]
            )

    def test_returns_empty_gdf_when_all_squares_missing(self, mocker):
        """When the layer file is missing from every requested square, an empty GeoDataFrame is returned."""
        mocker.patch(
            "geopandas.read_file",
            side_effect=DataSourceError(
                "'path' does not exist in the file system, and is not recognized as a supported dataset name."
            ),
        )
        gdf = load_geodata.load_gdf_os_openmap_layer(
            "important_building", grid_squares=["ND", "HX"]
        )
        assert gdf.empty
        assert "geometry" in gdf.columns
        assert gdf.crs.to_epsg() == 27700


class TestLoadGdfOsOpenroad:
    """Tests for `load_gdf_os_openroad`."""

    def test_skips_missing_square(self, mocker):
        """Squares whose road file is missing from S3 are skipped; the rest are returned."""
        mocker.patch(
            "geopandas.read_file", side_effect=read_file_missing_hx_side_effect
        )
        gdf = load_geodata.load_gdf_os_openroad(grid_squares=["HY", "HX"])
        assert len(gdf) == 1

    def test_reraises_other_data_source_errors(self, mocker):
        """DataSourceErrors unrelated to missing files (e.g. credentials) must not be skipped."""
        mocker.patch(
            "geopandas.read_file",
            side_effect=DataSourceError("AWS error: access denied"),
        )
        with pytest.raises(DataSourceError):
            load_geodata.load_gdf_os_openroad(grid_squares=["ND"])
