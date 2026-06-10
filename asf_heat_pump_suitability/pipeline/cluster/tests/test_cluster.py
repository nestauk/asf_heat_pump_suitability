import pytest
import geopandas as gpd
from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
    overlay_gdf_physical_barriers,
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

    def test_clusters_contain_domestic_only(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        # TODO requires adding clusters which should be removed (e.g. commercial)
        """Test that there are only domestic building footprints in the clusters (i.e. no non-domestic buildings are
        retained) and test that there are no clusters retained which do not contain a domestic building (i.e. no empty
        clusters)."""
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

        # Check only domestic building IDs are retained
        results = clusters_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
            "building_id"
        ]
        expected = buildings_gdf[buildings_gdf["type"] == "domestic"]

        assert set(results) == set(expected)

        # Check there are no empty clusters
        results = (
            clusters_gdf.sjoin(buildings_gdf, how="left", predicate="contains")[
                "building_id"
            ]
            .isna()
            .sum()
        )
        assert results == 0

    def test_clusters_not_overlapping(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        # TODO requires buildings that wrap around another building and groups of buildings that encase another building
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


class TestOverlayGdfPhysicalBarriers:
    @pytest.fixture(scope="class")
    def voronoi_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def tech_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def line_overlay_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def polygon_overlay_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def buildings_gdf(self):
        return gpd.GeoDataFrame()

    def test_cells_entirely_contain_buildings(
        self,
        voronoi_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        buildings_gdf,
    ):
        # TODO requires an edge case input where overlaying barriers remove all and some of the property
        """Test Voronoi cells entirely contain buildings after overlaying barriers. This tests the function can handle
        edge cases like a barrier completely or partially overlapping a building footprint and thus fragmenting the cell
        so that it no longer encases a full building footprint."""
        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
        )

        results = cells_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
            "building_id"
        ]
        expected = buildings_gdf["building_id"]

        assert set(results) == set(expected)
