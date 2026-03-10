"""Unit tests for asf_heat_pump_suitability.pipeline.transform.uprns."""

import geopandas as gpd
import polars as pl
from shapely.geometry import Point, Polygon

from asf_heat_pump_suitability.pipeline.transform import uprns


def make_df(records: list[dict]) -> pl.DataFrame:
    """Create a minimal Polars DataFrame from a list of dicts.

    Args:
        records: List of row dicts with at minimum UPRN, X_COORDINATE, Y_COORDINATE.

    Returns:
        pl.DataFrame: Polars DataFrame.
    """
    return pl.DataFrame(records)


def make_gdf(uprn_list: list[int], coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with UPRN and point geometries in BNG.

    Args:
        uprn_list: List of UPRN integers.
        coords: List of (x, y) BNG coordinate tuples.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with UPRN and geometry columns.
    """
    return gpd.GeoDataFrame(
        {"UPRN": uprn_list},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:27700",
    )


class TestGenerateGdfUprnCoords:
    """Tests for generate_gdf_uprn_coords."""

    def test_returns_geodataframe(self):
        """Output is a GeoDataFrame."""
        df = make_df([{"UPRN": 1, "X_COORDINATE": 100.0, "Y_COORDINATE": 200.0}])
        result = uprns.generate_gdf_uprn_coords(df)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_is_bng(self):
        """Output CRS is British National Grid (EPSG:27700)."""
        df = make_df([{"UPRN": 1, "X_COORDINATE": 100.0, "Y_COORDINATE": 200.0}])
        result = uprns.generate_gdf_uprn_coords(df)
        assert result.crs.to_epsg() == 27700

    def test_geometry_is_point(self):
        """Each row has a Point geometry."""
        df = make_df([{"UPRN": 1, "X_COORDINATE": 100.0, "Y_COORDINATE": 200.0}])
        result = uprns.generate_gdf_uprn_coords(df)
        assert result.geometry[0].geom_type == "Point"

    def test_coordinates_correct(self):
        """Point geometry has correct x and y coordinates."""
        df = make_df([{"UPRN": 1, "X_COORDINATE": 123.45, "Y_COORDINATE": 678.90}])
        result = uprns.generate_gdf_uprn_coords(df)
        pt = result.geometry[0]
        assert abs(pt.x - 123.45) < 1e-6
        assert abs(pt.y - 678.90) < 1e-6

    def test_multiple_rows(self):
        """Handles multiple rows correctly."""
        df = make_df(
            [
                {"UPRN": 1, "X_COORDINATE": 100.0, "Y_COORDINATE": 200.0},
                {"UPRN": 2, "X_COORDINATE": 300.0, "Y_COORDINATE": 400.0},
            ]
        )
        result = uprns.generate_gdf_uprn_coords(df)
        assert len(result) == 2

    def test_usecols_subset(self):
        """When usecols is specified, only those columns appear (plus coords)."""
        df = make_df(
            [
                {
                    "UPRN": 1,
                    "X_COORDINATE": 100.0,
                    "Y_COORDINATE": 200.0,
                    "EXTRA_COL": "drop_me",
                }
            ]
        )
        result = uprns.generate_gdf_uprn_coords(df, usecols=["UPRN"])
        assert "EXTRA_COL" not in result.columns


class TestFilterGdfResidentialUprns:
    """Tests for filter_gdf_residential_uprns."""

    def _make_building_polygon(self, cx: float, cy: float, r: float = 10.0) -> Polygon:
        """Create a square polygon centred on (cx, cy) with half-side r."""
        return Polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)])

    def test_uprn_in_building_not_non_residential_is_kept(self):
        """A UPRN inside a building that is not non-residential is retained."""
        uprn_gdf = make_gdf([1], [(50.0, 50.0)])
        buildings_gdf = gpd.GeoDataFrame(
            {"dummy": [1]},
            geometry=[self._make_building_polygon(50.0, 50.0)],
            crs="EPSG:27700",
        )
        non_res_gdf = gpd.GeoDataFrame(
            {"dummy": []},
            geometry=[],
            crs="EPSG:27700",
        )

        # Patch EPC loaders so no network calls are made
        import unittest.mock as mock

        with mock.patch(
            "asf_heat_pump_suitability.pipeline.transform.uprns.load_set_valid_epc_uprns",
            return_value=set(),
        ):
            result = uprns.filter_gdf_residential_uprns(
                uprn_gdf=uprn_gdf,
                buildings_gdf=buildings_gdf,
                non_residential_buildings_gdf=non_res_gdf,
            )

        assert 1 in result["UPRN"].values

    def test_uprn_in_non_residential_building_is_dropped(self):
        """A UPRN inside a non-residential building is excluded."""
        uprn_gdf = make_gdf([1], [(50.0, 50.0)])
        buildings_gdf = gpd.GeoDataFrame(
            {"dummy": [1]},
            geometry=[self._make_building_polygon(50.0, 50.0)],
            crs="EPSG:27700",
        )
        non_res_gdf = gpd.GeoDataFrame(
            {"dummy": [1]},
            geometry=[self._make_building_polygon(50.0, 50.0)],
            crs="EPSG:27700",
        )

        import unittest.mock as mock

        with mock.patch(
            "asf_heat_pump_suitability.pipeline.transform.uprns.load_set_valid_epc_uprns",
            return_value=set(),
        ):
            result = uprns.filter_gdf_residential_uprns(
                uprn_gdf=uprn_gdf,
                buildings_gdf=buildings_gdf,
                non_residential_buildings_gdf=non_res_gdf,
            )

        assert 1 not in result["UPRN"].values

    def test_uprn_in_domestic_epc_is_kept_regardless(self):
        """A UPRN in the domestic EPC register is kept even if not in any building."""
        uprn_gdf = make_gdf([99], [(1000.0, 1000.0)])  # far from any building
        empty_gdf = gpd.GeoDataFrame({"dummy": []}, geometry=[], crs="EPSG:27700")

        import unittest.mock as mock

        def mock_epc(epc_type):
            if epc_type == "domestic":
                return {99}
            return set()

        with mock.patch(
            "asf_heat_pump_suitability.pipeline.transform.uprns.load_set_valid_epc_uprns",
            side_effect=mock_epc,
        ):
            result = uprns.filter_gdf_residential_uprns(
                uprn_gdf=uprn_gdf,
                buildings_gdf=empty_gdf,
                non_residential_buildings_gdf=empty_gdf,
            )

        assert 99 in result["UPRN"].values
