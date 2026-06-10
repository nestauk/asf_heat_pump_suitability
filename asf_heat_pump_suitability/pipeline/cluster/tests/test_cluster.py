import pytest
import geopandas as gpd
from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
)


class TestGenerateGdfClusters:
    @pytest.fixture(scope="class")
    def buildings_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def boundary_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def tech_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def polygon_overlay_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def line_overlay_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def anchor_gdf(self):
        return gpd.GeoDataFrame()

    def test_all_domestic_buildings_assigned_cluster(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test each domestic building is assigned to a cluster."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )
        results = clusters_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
            "building_id"
        ]
        expected = buildings_gdf["building_id"]

        assert set(results) == set(expected)

    def test_clusters_not_overlapping(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test there are no overlapping clusters."""
        results = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )

        assert round(results["geometry"].area.sum(), 8) == round(
            results["geometry"].union_all().area, 8
        )

    def test_clusters_entirely_contain_buildings(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test clusters entirely contain buildings."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )

        results = clusters_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
            "building_id"
        ]
        expected = buildings_gdf["building_id"]

        assert set(results) == set(expected)

    def test_no_empty_clusters(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test there are no clusters that do not match to a building."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )

        results = (
            clusters_gdf.sjoin(buildings_gdf, how="left", predicate="contains")[
                "building_id"
            ]
            .isna()
            .sum()
        )
        assert results == 0

    def test_clusters_contain_domestic_only(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test that there are only domestic building footprints in the clusters."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )
        results = clusters_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
            "building_id"
        ]
        expected = buildings_gdf[buildings_gdf["type"] == "domestic"]

        assert set(results) == set(expected)

    def test_neighbouring_clusters_different(self):
        """Check dissolve worked properly"""
        pass


class TestReassignGdfAnchorProperties:
    def test_cells_within_anchor_radius(self):
        pass

    def test_cells_outside_anchor_radius(self):
        pass

    def test_cells_intersecting_anchor_radius(self):
        pass


class TestExtendEdgesGdf:
    def test_voronoi_contain_single_buildings(self):
        pass
