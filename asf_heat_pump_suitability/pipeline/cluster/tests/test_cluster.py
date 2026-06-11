import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import rotate
from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
    overlay_gdf_physical_barriers,
)
from asf_heat_pump_suitability.pipeline.cluster.tests import utils


class TestGenerateGdfClusters:
    @pytest.fixture(scope="class")
    def buildings_gdf(self):
        # Initialize lists to store geometries and IDs
        geometries = []
        building_ids = []

        # =====================================================================
        # CASE 1: Wrap-around building scenario
        # =====================================================================
        b01_inner = Polygon(
            [(400000, 399995), (400010, 399995), (400010, 400005), (400000, 400005)]
        )

        b02_wrap = Polygon(
            [
                (399990, 399990),
                (399990, 400015),
                (400020, 400015),
                (400020, 399990),
                (400015, 399990),
                (400015, 400010),
                (399995, 400010),
                (399995, 399990),
            ]
        )

        geometries.extend([b01_inner, b02_wrap])
        building_ids.extend(["B01_inner", "B02_wrap"])

        # =====================================================================
        # CASE 2: Small cluster surrounded by realistic building types
        # =====================================================================
        b03_cluster1 = Polygon(
            [(400060, 400000), (400070, 400000), (400070, 400010), (400060, 400010)]
        )
        b04_cluster2 = Polygon(
            [
                (400075, 399995),
                (400085, 399995),
                (400085, 400005),
                (400080, 400005),
                (400080, 400010),
                (400075, 400010),
            ]
        )

        geometries.extend([b03_cluster1, b04_cluster2])
        building_ids.extend(["B03_cluster_1", "B04_cluster_2"])

        # Surrounding buildings (outer ring)
        b05_base = Polygon(
            [(400055, 400025), (400090, 400025), (400090, 400030), (400055, 400030)]
        )
        b05_rotated = rotate(b05_base, 15, origin="center")

        b06_block = Polygon(
            [
                (400100, 400015),
                (400110, 400015),
                (400110, 400020),
                (400105, 400020),
                (400105, 400025),
                (400100, 400025),
            ]
        )

        b07_base = Polygon(
            [(400105, 399990), (400120, 399990), (400120, 400010), (400105, 400010)]
        )
        b07_rotated = rotate(b07_base, -20, origin="center")

        b08_z_shape = Polygon(
            [
                (400090, 399985),
                (400110, 399985),
                (400110, 399980),
                (400102, 399980),
                (400102, 399972),
                (400115, 399972),
                (400115, 399967),
                (400095, 399967),
                (400095, 399972),
                (400097, 399972),
                (400097, 399980),
                (400090, 399980),
            ]
        )

        b09_strip = Polygon(
            [(400060, 399975), (400085, 399975), (400085, 399980), (400060, 399980)]
        )
        b10_irreg = Polygon(
            [
                (400040, 399975),
                (400050, 399975),
                (400050, 399985),
                (400045, 399988),
                (400040, 399985),
            ]
        )

        b11_terrace = Polygon(
            [
                (400030, 399990),
                (400030, 400010),
                (400040, 400010),
                (400040, 400008),
                (400043, 400008),
                (400043, 400005),
                (400040, 400005),
                (400040, 399995),
                (400043, 399995),
                (400043, 399992),
                (400040, 399992),
                (400040, 399990),
            ]
        )

        b12_horseshoe = Polygon(
            [
                (400030, 400015),
                (400050, 400015),
                (400050, 400020),
                (400040, 400020),
                (400040, 400030),
                (400050, 400030),
                (400050, 400035),
                (400030, 400035),
            ]
        )

        surrounding_shapes = [
            b05_rotated,
            b06_block,
            b07_rotated,
            b08_z_shape,
            b09_strip,
            b10_irreg,
            b11_terrace,
            b12_horseshoe,
        ]
        geometries.extend(surrounding_shapes)
        building_ids.extend([f"B{i:02d}_surround" for i in range(5, 13)])

        # Build the base buildings GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"building_id": building_ids, "geometry": geometries}, crs="EPSG:27700"
        )

        return gdf

    @pytest.fixture(scope="class")
    def boundary_gdf(self, buildings_gdf):
        # Generate site boundary with a 12m buffer
        combined_footprints = buildings_gdf.union_all()
        site_boundary_geom = combined_footprints.buffer(12).convex_hull

        return gpd.GeoDataFrame(
            {"name": ["SITE_BOUNDARY"], "geometry": [site_boundary_geom]},
            crs="EPSG:27700",
        )

    @pytest.fixture(scope="class")
    def gdf_tech_mixed_domestic(self, buildings_gdf):
        non_domestic_ids = ["B03", "B05", "B08"]

        tech_mapping = {
            "B01": "Individual solution",
            "B05": "Individual solution",
            "B06": "Individual solution",
            "B07": "Communal solution",
            "B11": "Communal solution",
            "B12": "Communal solution",
            "B02": "Networked heat pump",
            "B03": "Networked heat pump",
            "B04": "Networked heat pump",
            "B10": "Individual solution",
            "B08": "District heat network",
            "B09": "District heat network",
        }

        # Apply functions across the dataframe
        buildings_gdf["domestic"] = buildings_gdf["building_id"].apply(
            utils.assign_domestic_status, args=(non_domestic_ids,)
        )
        buildings_gdf["assigned_tech"] = buildings_gdf["building_id"].apply(
            utils.assign_tech_type, args=(tech_mapping,)
        )

        return buildings_gdf

    @pytest.fixture(scope="class")
    def polygon_overlay_gdf(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    @pytest.fixture(scope="class")
    def line_overlay_gdf(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    @pytest.fixture(scope="class")
    def anchor_gdf(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    def test_all_buildings_assigned_cluster(
        self,
        buildings_gdf,
        boundary_gdf,
        gdf_tech_mixed_domestic,
        line_overlay_gdf,
        polygon_overlay_gdf,
        anchor_gdf,
    ):
        """Test each domestic building is assigned to a cluster."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=gdf_tech_mixed_domestic,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=anchor_gdf,
            radius=50,
            id_col="building_id",
        )

        # Check if the uncontained area is effectively zero (e.g., less than 1 square millimeter)
        # This is to account for floating point errors which can cause tiny slivers of building not to be covered by
        # the cluster.
        uncontained_slivers = gpd.overlay(
            buildings_gdf, clusters_gdf, how="difference", keep_geom_type=False
        )
        uncontained_slivers["area"] = uncontained_slivers["geometry"].area
        results = uncontained_slivers[uncontained_slivers["area"] > 1e-5]
        assert (
            len(results) == 0
        ), f"Buildings {set(results['building_id'])} are missing significant coverage in the clustering."


#
#     def test_clusters_contain_domestic_only(
#         self,
#         buildings_gdf,
#         boundary_gdf,
#         tech_gdf,
#         line_overlay_gdf,
#         polygon_overlay_gdf,
#         anchor_gdf,
#     ):
#         # TODO requires adding clusters which should be removed (e.g. commercial)
#         """Test that there are only domestic building footprints in the clusters (i.e. no non-domestic buildings are
#         retained) and test that there are no clusters retained which do not contain a domestic building (i.e. no empty
#         clusters)."""
#         clusters_gdf = generate_gdf_clusters(
#             buildings_gdf=buildings_gdf,
#             boundary_gdf=boundary_gdf,
#             tech_gdf=tech_gdf,
#             line_overlay_gdf=line_overlay_gdf,
#             polygon_overlay_gdf=polygon_overlay_gdf,
#             combined_anchor_gdf=anchor_gdf,
#             radius=50,
#             id_col="building_id",
#         )
#
#         # Check only domestic building IDs are retained
#         results = clusters_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
#             "building_id"
#         ]
#         expected = buildings_gdf[buildings_gdf["type"] == "domestic"]
#
#         assert set(results) == set(expected)
#
#         # Check there are no empty clusters
#         results = (
#             clusters_gdf.sjoin(buildings_gdf, how="left", predicate="contains")[
#                 "building_id"
#             ]
#             .isna()
#             .sum()
#         )
#         assert results == 0
#
#     def test_clusters_not_overlapping(
#         self,
#         buildings_gdf,
#         boundary_gdf,
#         tech_gdf,
#         line_overlay_gdf,
#         polygon_overlay_gdf,
#         anchor_gdf,
#     ):
#         # TODO requires buildings that wrap around another building and groups of buildings that encase another building
#         """Test there are no overlapping clusters."""
#         results = generate_gdf_clusters(
#             buildings_gdf=buildings_gdf,
#             boundary_gdf=boundary_gdf,
#             tech_gdf=tech_gdf,
#             line_overlay_gdf=line_overlay_gdf,
#             polygon_overlay_gdf=polygon_overlay_gdf,
#             combined_anchor_gdf=anchor_gdf,
#             radius=50,
#             id_col="building_id",
#         )
#
#         assert round(results["geometry"].area.sum(), 8) == round(
#             results["geometry"].union_all().area, 8
#         )
#
#     def test_touching_neighbours(self):
#         pass
#
#     def test_buildings_contained_by_others(self):
#         pass
#
#     def test_neighbouring_clusters_different(self):
#         """Check dissolve worked properly"""
#         pass
#
#
# class TestReassignGdfAnchorProperties:
#     def test_cells_within_anchor_radius(self):
#         pass
#
#     def test_cells_outside_anchor_radius(self):
#         pass
#
#     def test_cells_intersecting_anchor_radius(self):
#         pass
#
#
# class TestExtendEdgesGdf:
#     @pytest.fixture(scope="class")
#     def polygon_gdf(self):
#         return gpd.GeoDataFrame()
#
#     @pytest.fixture(scope="class")
#     def boundary(self):
#         return gpd.GeoDataFrame()
#
#     def test_voronoi_contain_single_polygons(self, polygon_gdf, boundary):
#         cells_gdf = extend_edges_gdf(gdf=polygon_gdf, boundary=boundary)
#         results = cells_gdf.sjoin(polygon_gdf, how="inner", predicate="contains")
#         assert len(results) == len(polygon_gdf)
#         assert set(results["id"]) == set(polygon_gdf["id"])
#
#     def test_polygons_within_boundary(self, polygon_gdf, boundary):
#         cells_gdf = extend_edges_gdf(gdf=polygon_gdf, boundary=boundary)
#         results = cells_gdf.sjoin(polygon_gdf, how="inner", predicate="contains")["id"]
#         expected = polygon_gdf[polygon_gdf["within_boundary"]]["id"]
#         assert set(results) == set(expected)
#
#     def test_voronoi_larger_than_buffer(self):
#         pass
#
#
# class TestOverlayGdfPhysicalBarriers:
#     @pytest.fixture(scope="class")
#     def voronoi_gdf(self):
#         return gpd.GeoDataFrame()
#
#     @pytest.fixture(scope="class")
#     def tech_gdf(self):
#         return gpd.GeoDataFrame()
#
#     @pytest.fixture(scope="class")
#     def line_overlay_gdf(self):
#         return gpd.GeoDataFrame()
#
#     @pytest.fixture(scope="class")
#     def polygon_overlay_gdf(self):
#         return gpd.GeoDataFrame()
#
#     @pytest.fixture(scope="class")
#     def buildings_gdf(self):
#         return gpd.GeoDataFrame()
#
#     def test_cells_entirely_contain_buildings(
#         self,
#         voronoi_gdf,
#         tech_gdf,
#         line_overlay_gdf,
#         polygon_overlay_gdf,
#         buildings_gdf,
#     ):
#         # TODO requires an edge case input where overlaying barriers remove all and some of the property (including overlapping and bisecting)
#         # TODO edge case where original Voronoi cell doesn't completely cover building
#         """Test Voronoi cells entirely contain buildings after overlaying barriers. This tests the function can handle
#         edge cases like a barrier completely or partially overlapping a building footprint and thus fragmenting the cell
#         so that it no longer encases a full building footprint."""
#         cells_gdf = overlay_gdf_physical_barriers(
#             voronoi_gdf=voronoi_gdf,
#             tech_gdf=tech_gdf,
#             line_overlay_gdf=line_overlay_gdf,
#             polygon_overlay_gdf=polygon_overlay_gdf,
#         )
#
#         results = cells_gdf.sjoin(buildings_gdf, how="inner", predicate="contains")[
#             "building_id"
#         ]
#         expected = buildings_gdf["building_id"]
#
#         assert set(results) == set(expected)
