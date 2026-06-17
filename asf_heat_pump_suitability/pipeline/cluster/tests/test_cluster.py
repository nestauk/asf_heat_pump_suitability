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


@pytest.fixture(scope="module")
def buildings_gdf():
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

    # =====================================================================
    # CASE 3: TIGHT EDGE CASES (B13 & B14)
    # =====================================================================
    # B13: L-shaped footprint directly south of B01/B02, 2m away from B02's base
    b13_l_encase = Polygon(
        [
            (399990, 399970),
            (400000, 399970),
            (400000, 399982),
            (400020, 399982),
            (400020, 399988),
            (399990, 399988),
        ]
    )

    # B14: Footprint positioned exactly 1m east of B13's vertical wall
    b14_small_gap = Polygon(
        [
            (400021, 399982),
            (400029, 399982),
            (400029, 399988),
            (400021, 399988),
        ]
    )

    geometries = [
        b01_inner,
        b02_wrap,
        b03_cluster1,
        b04_cluster2,
        b05_rotated,
        b06_block,
        b07_rotated,
        b08_z_shape,
        b09_strip,
        b10_irreg,
        b11_terrace,
        b12_horseshoe,
        b13_l_encase,
        b14_small_gap,
    ]
    building_ids = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
    ]

    # Build the base buildings GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"building_id": building_ids, "geometry": geometries}, crs="EPSG:27700"
    )

    return gdf


@pytest.fixture(scope="module")
def boundary_gdf(buildings_gdf):
    # Generate site boundary with a 12m buffer
    combined_footprints = buildings_gdf.union_all()
    site_boundary_geom = combined_footprints.buffer(12).convex_hull

    return gpd.GeoDataFrame(
        {"name": ["SITE_BOUNDARY"], "geometry": [site_boundary_geom]},
        crs="EPSG:27700",
    )


class TestGenerateGdfClusters:
    @pytest.fixture(scope="class")
    def tech_gdf(self, buildings_gdf):
        non_domestic_ids = ["B03", "B05", "B08"]

        tech_mapping = {
            "B01": "Individual solution",
            "B05": "Individual solution",
            "B06": "Individual solution",
            "B10": "Individual solution",
            "B14": "Individual solution",
            "B07": "Communal solution",
            "B11": "Communal solution",
            "B12": "Communal solution",
            "B13": "Communal solution",
            "B02": "Networked heat pump",
            "B03": "Networked heat pump",
            "B04": "Networked heat pump",
            "B08": "District heat network",
            "B09": "District heat network",
        }

        # Apply functions across the dataframe
        buildings_gdf["domestic"] = buildings_gdf["building_id"].apply(
            utils.assign_bool_domestic_status, args=(non_domestic_ids,)
        )
        buildings_gdf["assigned_tech"] = buildings_gdf["building_id"].apply(
            utils.assign_str_tech_type, args=(tech_mapping,)
        )

        return buildings_gdf

    @pytest.fixture(scope="class")
    def polygon_overlay_gdf(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    @pytest.fixture(scope="class")
    def line_overlay_gdf(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    @pytest.fixture(scope="class")
    def gdf_empty_anchors(self):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    @pytest.fixture(scope="class")
    def gdf_anchor_properties(self, buildings_gdf):
        non_domestic_ids = ["B03", "B05", "B08"]

        # Apply functions across the dataframe
        buildings_gdf["domestic"] = buildings_gdf["building_id"].apply(
            utils.assign_bool_domestic_status, args=(non_domestic_ids,)
        )

        return buildings_gdf[~buildings_gdf["domestic"]].copy()

    def test_all_buildings_assigned_cluster(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        gdf_empty_anchors,
    ):
        """Test each domestic building is assigned to a cluster."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=gdf_empty_anchors,
            radius=50,
            id_col="building_id",
        )

        # Check if the uncontained area is effectively zero (e.g., less than 1 square millimeter)
        # This is to account for floating point errors which can cause tiny slivers of building not to be covered by
        # the cluster.
        uncontained_slivers_gdf = buildings_gdf.overlay(
            clusters_gdf, how="difference", keep_geom_type=False
        )
        uncontained_slivers_gdf["area"] = uncontained_slivers_gdf["geometry"].area
        results = uncontained_slivers_gdf[uncontained_slivers_gdf["area"] > 1e-5]
        assert (
            len(results) == 0
        ), f"Buildings {set(results['building_id'])} are missing significant coverage in the clustering."

    def test_clusters_contain_domestic_only(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        gdf_empty_anchors,
    ):
        """Test that there are only domestic building footprints in the clusters (i.e. no non-domestic buildings are
        retained) and test that there are no clusters retained which do not contain a domestic building (i.e. no empty
        clusters)."""
        domestic_tech_gdf = tech_gdf[tech_gdf["domestic"]].copy()

        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=domestic_tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=gdf_empty_anchors,
            radius=50,
            id_col="building_id",
        )

        # Check only domestic building IDs are retained
        non_domestic_gdf = tech_gdf[~tech_gdf["domestic"]]
        results = clusters_gdf.sjoin(
            non_domestic_gdf, how="inner", predicate="intersects"
        )["building_id"]

        assert (
            len(results) == 0
        ), "Some non-domestic building footprints are found in the clusters."

        # Check there are no empty clusters
        empty_clusters_gdf = buildings_gdf.overlay(
            clusters_gdf, how="intersection", keep_geom_type=False
        ).dissolve(by="cluster_id")
        empty_clusters_gdf["area"] = empty_clusters_gdf["geometry"].area

        # Find smallest building area and subtract small sliver for floating point errors
        smallest_building_area = buildings_gdf.area.min() - 1e-5

        # Clusters must intersect with at least the area of the smallest building footprint, otherwise they are considered empty
        results = empty_clusters_gdf[
            empty_clusters_gdf["area"] <= smallest_building_area
        ]
        assert (
            len(results) == 0
        ), "There are clusters which do not contain any buildings."

    def test_clusters_not_overlapping(
        self,
        buildings_gdf,
        boundary_gdf,
        tech_gdf,
        line_overlay_gdf,
        polygon_overlay_gdf,
        gdf_empty_anchors,
    ):
        """Test there are no overlapping clusters."""
        results = generate_gdf_clusters(
            buildings_gdf=buildings_gdf,
            boundary_gdf=boundary_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            combined_anchor_gdf=gdf_empty_anchors,
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
    @pytest.fixture(scope="class")
    def gdf_polygons_across_boundary(self):
        return gpd.GeoDataFrame(
            {
                "building_id": ["B01", "B02", "B03", "B04", "B_OUTSIDE", "B_CROSSING"],
                "within_boundary": [True, True, True, True, False, False],
                "geometry": [
                    Polygon(
                        [
                            (400000, 399995),
                            (400010, 399995),
                            (400010, 399995),
                            (400000, 400005),
                        ]
                    ),
                    Polygon(
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
                    ),
                    Polygon(
                        [
                            (400060, 399900),
                            (400070, 399900),
                            (400070, 399910),
                            (400060, 399910),
                        ]
                    ),
                    Polygon(
                        [
                            (400075, 399995),
                            (400085, 399995),
                            (400085, 399905),
                            (400080, 400005),
                            (400080, 400010),
                            (400075, 400010),
                        ]
                    ),
                    # Outside: Y=400035 to 400045
                    Polygon(
                        [
                            (400070, 400035),
                            (400080, 400035),
                            (400080, 400045),
                            (400070, 400045),
                        ]
                    ),
                    # Crossing: Y=400025 to 400035
                    Polygon(
                        [
                            (400050, 400025),
                            (400060, 400025),
                            (400060, 400035),
                            (400050, 400035),
                        ]
                    ),
                ],
            },
            crs="EPSG:27700",
        )

    @pytest.fixture(scope="class")
    def geometry_boundary_crossed(self):
        return Polygon(
            [
                (399978.0, 399880.0),
                (400100.0, 399880.0),
                (400100.0, 400030.0),
                (399978.0, 400030.0),
            ]
        )

    def test_voronoi_contain_single_polygons(self, buildings_gdf, boundary_gdf):
        """Test one Voronoi polygon contains one building footprint."""
        boundary = boundary_gdf.geometry.iloc[0]
        cells_gdf = extend_edges_gdf(gdf=buildings_gdf, boundary=boundary)
        # All buildings should match to a cell
        assert set(cells_gdf["building_id"]) == set(
            buildings_gdf["building_id"]
        ), f"Unique building IDs in buildings and Voronoi cells do not match"

        cells_gdf["geometry"] = cells_gdf["geometry"].make_valid().normalize()
        # Buildings and cells should have a 1-1 mapping
        assert (
            cells_gdf["building_id"].duplicated().sum() == 0
        ), f"{cells_gdf['building_id'].duplicated().sum()} building IDs joined to multiple cells"
        assert (
            cells_gdf["geometry"].duplicated().sum() == 0
        ), f"{cells_gdf['geometry'].duplicated().sum()} cells joined to multiple buildings"

    def test_polygons_within_boundary(
        self, gdf_polygons_across_boundary, geometry_boundary_crossed
    ):
        cells_gdf = extend_edges_gdf(
            gdf=gdf_polygons_across_boundary, boundary=geometry_boundary_crossed
        )
        print(cells_gdf.columns)
        results = cells_gdf["building_id"]
        expected = gdf_polygons_across_boundary[
            gdf_polygons_across_boundary["within_boundary"]
        ]["building_id"]
        assert set(results) == set(expected)


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
