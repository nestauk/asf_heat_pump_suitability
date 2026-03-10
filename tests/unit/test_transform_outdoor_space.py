"""Unit tests for asf_heat_pump_suitability.pipeline.transform.outdoor_space."""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Polygon

from asf_heat_pump_suitability.pipeline.transform import outdoor_space


def square(cx: float, cy: float, r: float) -> Polygon:
    """Create a square polygon centred on (cx, cy) with half-side r.

    Args:
        cx: Centre x coordinate.
        cy: Centre y coordinate.
        r: Half-side length.

    Returns:
        Polygon: Square polygon.
    """
    return Polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)])


class TestGenerateGdfBuildingIntersections:
    """Tests for generate_gdf_building_intersections."""

    def test_intersection_within_min_size_kept(self):
        """Building intersections above minimum area threshold are retained."""
        land = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["L1"]},
            geometry=[square(50, 50, 50)],
            crs="EPSG:27700",
        )
        building = gpd.GeoDataFrame(
            {"ID": ["B1"]},
            geometry=[square(50, 50, 20)],  # large building, fully inside land
            crs="EPSG:27700",
        )
        result = outdoor_space.generate_gdf_building_intersections(
            land_parcels_gdf=land, building_footprints_gdf=building
        )
        assert len(result) >= 1

    def test_tiny_corner_intersection_filtered(self):
        """Tiny corner intersections below minimum area threshold are dropped."""
        land = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["L1"]},
            geometry=[square(0, 0, 1)],  # small 2x2 metre land parcel
            crs="EPSG:27700",
        )
        # Large building that just barely touches the land parcel in a tiny corner
        building = gpd.GeoDataFrame(
            {"ID": ["B1"]},
            geometry=[square(100, 100, 90)],  # large building, mostly outside land
            crs="EPSG:27700",
        )
        result = outdoor_space.generate_gdf_building_intersections(
            land_parcels_gdf=land,
            building_footprints_gdf=building,
            min_intersection=15,  # intersection must be at least 15m2
        )
        assert len(result) == 0


class TestGenerateGdfOutdoorSpace:
    """Tests for generate_gdf_outdoor_space."""

    def test_outdoor_space_is_land_minus_building(self):
        """Outdoor space area equals land area minus building intersection area."""
        land = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["L1"]},
            geometry=[square(50, 50, 50)],  # 100m x 100m = 10000 m2
            crs="EPSG:27700",
        )
        # Building intersection occupies 20m x 20m = 400m2
        intersection = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["L1"], "building_area_m2": [400.0]},
            geometry=[square(50, 50, 10)],
            crs="EPSG:27700",
        )
        result = outdoor_space.generate_gdf_outdoor_space(
            building_intersections_gdf=intersection, land_parcels_gdf=land
        )
        assert len(result) == 1
        row = result.iloc[0]
        # Outdoor space ~ 10000 - 400 = 9600 m2 (approximately)
        assert row["total_outdoor_space_area_m2"] > 9000

    def test_no_buildings_gives_full_land_area(self):
        """When there are no building intersections, all land is outdoor space."""
        land = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["L1"]},
            geometry=[square(50, 50, 50)],  # 10000 m2
            crs="EPSG:27700",
        )
        empty_intersection = gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": [], "building_area_m2": []},
            geometry=[],
            crs="EPSG:27700",
        )
        result = outdoor_space.generate_gdf_outdoor_space(
            building_intersections_gdf=empty_intersection, land_parcels_gdf=land
        )
        assert len(result) == 1
        assert result.iloc[0]["total_outdoor_space_area_m2"] == pytest.approx(10000, rel=0.01)


class TestDeduplicateDfOutdoorSpace:
    """Tests for deduplicate_df_outdoor_space."""

    def test_no_duplicates_unchanged(self):
        """DataFrame with no duplicate UPRNs is returned unchanged."""
        df = pl.DataFrame(
            {
                "UPRN": [1, 2, 3],
                "total_outdoor_space_area_m2": [100.0, 200.0, 300.0],
                "max_contiguous_outdoor_space_area_m2": [90.0, 190.0, 290.0],
            }
        )
        result = outdoor_space.deduplicate_df_outdoor_space(df)
        assert len(result) == 3

    def test_duplicate_uprn_kept_with_smallest_outdoor_space(self):
        """When a UPRN appears multiple times, the row with the smallest total area is kept."""
        df = pl.DataFrame(
            {
                "UPRN": [1, 1, 2],
                "total_outdoor_space_area_m2": [500.0, 100.0, 200.0],
                "max_contiguous_outdoor_space_area_m2": [400.0, 80.0, 150.0],
            }
        )
        result = outdoor_space.deduplicate_df_outdoor_space(df)
        uprn1 = result.filter(pl.col("UPRN") == 1)
        assert len(uprn1) == 1
        assert uprn1["total_outdoor_space_area_m2"][0] == pytest.approx(100.0)

    def test_returns_polars_dataframe(self):
        """Return type is always a Polars DataFrame."""
        df = pl.DataFrame(
            {
                "UPRN": [1],
                "total_outdoor_space_area_m2": [50.0],
                "max_contiguous_outdoor_space_area_m2": [40.0],
            }
        )
        result = outdoor_space.deduplicate_df_outdoor_space(df)
        assert isinstance(result, pl.DataFrame)
