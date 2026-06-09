from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
)


class TestGenerateGdfClusters:
    def test_clusters_not_overlapping(self):
        pass

    def test_clusters_entirely_contain_buildings(self):
        pass

    def test_no_empty_clusters(self):
        pass

    def test_clusters_contain_domestic_only(self):
        pass

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
