"""
Tests for grid-square getters in getters/load_geodata.py.

Genuine missing grid-square files on S3 (due to true absence of data e.g. sea-only squares
with no road layer) raise pyogrio DataSourceError and must be skipped, while any other read
failure must still raise.

The partitioned parquet getters (OS Open UPRN, POI, and the NSUL lookup in load_data.py) skip
a missing file only for a non-land square, and raise for a missing land-square file.
"""

import geopandas as gpd
import polars as pl
import pytest
from pyogrio.errors import DataSourceError
from shapely.geometry import Point

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_data, load_geodata

SAMPLE_GDF = gpd.GeoDataFrame(
    {"ID": [1], "geometry": [Point(0, 0)]}, crs=config["constant"]["target_crs"]
)


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


PARTITIONED_GETTERS = [  # format: (getter, config_key)
    (load_geodata.load_df_osopen_uprn, "uk_osopen_uprn_partitioned"),
    (load_geodata.load_gdf_poi, "UK_poi_locations_partitioned"),
    (load_data.load_df_uprn_lookup, "gb_uprn_lookup_partitioned"),
]


@pytest.mark.parametrize(
    "getter, config_key",
    PARTITIONED_GETTERS,
    ids=["osopen_uprn", "poi", "uprn_lookup"],
)
class TestPartitionedParquetGetters:
    """The partitioned getters skip a missing non-land square file and raise for a missing land-square file."""

    @staticmethod
    def mock_s3(mocker, config_key: str, squares_on_s3: list) -> object:
        """
        Set up a fake S3 for one test, so that only `squares_on_s3` have a partition file.

        It patches three things:

        - The land squares, to SE, NZ and SX, so that the tests do not depend on the config list.
          SX is never requested, so a test never asks for all the land squares. If a test did, the UPRN
          getter would read its single GB-wide file in place of the per-square files, and the missing-file
          check would not run.
        - The S3 folder listing, so that it returns only the files for `squares_on_s3`.
        - `pl.read_parquet`, so that the test can check which paths the getter read.

        Args:
            mocker (pytest_mock.MockerFixture): pytest-mock fixture.
            config_key (str): key of the partition path template in `config["data"]["geodata"]`.
            squares_on_s3 (list): grid squares that have a partition file.

        Returns:
            unittest.mock.MagicMock: the stubbed `pl.read_parquet`, to inspect the paths it read.
        """
        mocker.patch.dict(config["constant"], {"land_grid_squares": ["SE", "NZ", "SX"]})
        template = config["data"]["geodata"][config_key]
        mocker.patch("s3fs.S3FileSystem").return_value.ls.return_value = [
            template.format(grid_square=sq).removeprefix("s3://")
            for sq in squares_on_s3
        ]
        return mocker.patch(
            "polars.read_parquet",
            return_value=pl.DataFrame({"UPRN": [1], "PCDS": ["PL1 1AA"]}),
        )

    def test_skips_missing_non_land_square(self, mocker, getter, config_key):
        """A sea square (OV) with no file is skipped and the land square is still read."""
        read_parquet = self.mock_s3(mocker, config_key, squares_on_s3=["SE"])
        getter(grid_squares=["SE", "OV"])
        paths_read = str(read_parquet.call_args_list)
        assert "SE_" in paths_read, "the land square with a file must still be read"
        assert (
            "OV_" not in paths_read
        ), "a sea square with no file must be skipped, not read (it gives a 404)"

    def test_raises_for_missing_land_square(self, mocker, getter, config_key):
        """A land square (NZ) with no file is a data gap and must stop the run."""
        self.mock_s3(mocker, config_key, squares_on_s3=["SE"])
        with pytest.raises(FileNotFoundError, match="NZ"):
            getter(grid_squares=["SE", "NZ"])
