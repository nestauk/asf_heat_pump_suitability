"""
Tests for asf_heat_pump_suitability.pipeline.transform.outdoor_space.

Run:
pytest asf_heat_pump_suitability/pipeline/transform/tests/test_outdoor_space.py
"""

import geopandas as gpd
import pytest
from shapely.geometry import box

from asf_heat_pump_suitability.pipeline.transform.outdoor_space import (
    clip_gdf_land_parcels,
)


class TestClipGdfLandParcels:
    """Tests for `clip_gdf_land_parcels`."""

    @pytest.fixture(scope="class")
    def land_parcels_gdf(self):
        """
        Two land parcels: P1 is a 10m x 10m square which will be split by a barrier; P2 is a 10m x 10m square with
        no barrier running through it.
        """
        return gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["P1", "P2"]},
            geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
            crs="EPSG:27700",
        )

    @pytest.fixture(scope="class")
    def polygon_overlay_gdf(self):
        """A single barrier (e.g. a road) running east-west through the middle of P1. P2 is untouched."""
        return gpd.GeoDataFrame(
            {"barrier_id": ["road"]},
            geometry=[box(0, 4.5, 10, 5.5)],
            crs="EPSG:27700",
        )

    @pytest.fixture(scope="class")
    def intersection_gdf(self):
        """
        Building intersections: one in the north fragment of P1 (the building only touches the north half once
        the barrier splits the parcel), and one in P2.
        """
        return gpd.GeoDataFrame(
            {"NATIONALCADASTRALREFERENCE": ["P1", "P2"]},
            geometry=[box(2, 6, 4, 8), box(22, 2, 24, 4)],
            crs="EPSG:27700",
        )

    def test_retains_only_fragment_touching_building(
        self, land_parcels_gdf, intersection_gdf, polygon_overlay_gdf
    ):
        """Test that only the land parcel fragment touching a building intersection is retained, and the fragment
        on the other side of the barrier (with no building) is discarded."""
        results = clip_gdf_land_parcels(
            land_parcels_gdf, intersection_gdf, polygon_overlay_gdf
        )

        # P1 should be clipped down to just the north fragment (10m x 4.5m), not the full 10m x 10m parcel
        expected_p1_area = 10 * 4.5
        assert results.loc["P1", "geometry"].area == pytest.approx(
            expected_p1_area
        ), "P1 was not clipped down to the fragment touching the building intersection"

        # P2 has no barrier, so it should be returned in full
        expected_p2_area = 10 * 10
        assert results.loc["P2", "geometry"].area == pytest.approx(
            expected_p2_area
        ), "P2 area changed despite having no barrier running through it"

    def test_result_indexed_by_land_parcel_id(
        self, land_parcels_gdf, intersection_gdf, polygon_overlay_gdf
    ):
        """Test the returned GeoDataFrame is dissolved by, and indexed on, the land parcel ID with one row per
        parcel."""
        results = clip_gdf_land_parcels(
            land_parcels_gdf, intersection_gdf, polygon_overlay_gdf
        )

        assert results.index.name == "NATIONALCADASTRALREFERENCE"
        assert sorted(results.index) == [
            "P1",
            "P2",
        ], "Expected one row per land parcel, indexed by land parcel ID"
