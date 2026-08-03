"""
Unit tests for functions in cluster.py
"""

import pytest
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from shapely.affinity import rotate
from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
)
from asf_heat_pump_suitability.pipeline.cluster.tests import utils


@pytest.fixture(scope="module")
def gdf_mixed_buildings():
    """
    Generate a geodataframe containing a selection of test building footprints in BNG CRS (EPSG: 27700) to test clustering
    across different scenarios:
    1. A horseshoe-shaped building that wraps around another smaller building on three sides.
    2. A central cluster of buildings surrounded on all sides by a selection of buildings of different shapes.
    3. Buildings which are very close to the neighbouring buildings.
    """
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
    # CASE 2: Small cluster of buildings surrounded by different building types
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
    # CASE 3: BUILDINGS WHICH ARE VERY CLOSE TOGETHER
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

    gdf = gpd.GeoDataFrame(
        {"building_id": building_ids, "geometry": geometries}, crs="EPSG:27700"
    )

    return gdf


@pytest.fixture(scope="module")
def gdf_enclosing_boundary(gdf_mixed_buildings):
    """
    Generate a boundary geodataframe with a 12m buffer (slightly above 10m desired minimum distance between edges of any
    buildings and the edge of the map).
    """
    combined_footprints = gdf_mixed_buildings.union_all()
    site_boundary_geom = combined_footprints.buffer(12).convex_hull

    return gpd.GeoDataFrame(
        {"name": ["SITE_BOUNDARY"], "geometry": [site_boundary_geom]},
        crs="EPSG:27700",
    )


class TestGenerateGdfClusters:
    """Test the `generate_gdf_clusters` function."""

    @pytest.fixture(scope="class")
    def tech_gdf(self, gdf_mixed_buildings):
        """
        Assign building footprints a technology type and a boolean to indicate whether or not they are a domestic building.
        """
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

        # Assign domestic boolean to buildings according to mapping
        gdf_mixed_buildings["domestic"] = gdf_mixed_buildings["building_id"].apply(
            utils.assign_bool_domestic_status, args=(non_domestic_ids,)
        )

        # Assign tech type according to mapping
        gdf_mixed_buildings["assigned_tech"] = gdf_mixed_buildings["building_id"].apply(
            utils.assign_str_tech_type, args=(tech_mapping,)
        )

        return gdf_mixed_buildings

    @pytest.fixture(scope="class")
    def empty_gdf(self):
        """Create an empty geodataframe."""
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)

    def test_all_buildings_assigned_cluster(
        self,
        gdf_mixed_buildings,
        gdf_enclosing_boundary,
        tech_gdf,
        empty_gdf,
    ):
        """Test each domestic building is assigned to a cluster and only to one cluster."""
        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=gdf_mixed_buildings,
            boundary_gdf=gdf_enclosing_boundary,
            tech_gdf=tech_gdf,
            line_overlay_gdf=empty_gdf,
            polygon_overlay_gdf=empty_gdf,
            combined_anchor_gdf=empty_gdf,
            radius=50,
            id_col="building_id",
        )

        results = clusters_gdf[["geometry"]].sjoin(
            gdf_mixed_buildings, how="inner", predicate="contains"
        )["building_id"]

        # Assert each building is assigned to a cluster
        expected = gdf_mixed_buildings["building_id"]
        missing = set(expected).difference(set(results))
        assert (
            not missing
        ), f"Some buildings not contained by a cluster. Building IDs: {missing}"

        # Assert each building is assigned to only one cluster
        duplicated = results.duplicated()
        assert (
            duplicated.sum() == 0
        ), f"Some buildings are contained by multiple clusters. Building IDs: {results[duplicated].unique()}"

    def test_clusters_contain_domestic_only(
        self,
        gdf_mixed_buildings,
        gdf_enclosing_boundary,
        tech_gdf,
        empty_gdf,
    ):
        """Test that there are only domestic building footprints in the clusters (i.e. no non-domestic buildings are
        retained) and test that there are no clusters retained which do not contain a domestic building (i.e. no empty
        clusters)."""
        domestic_tech_gdf = tech_gdf[tech_gdf["domestic"]].copy()

        clusters_gdf = generate_gdf_clusters(
            buildings_gdf=gdf_mixed_buildings,
            boundary_gdf=gdf_enclosing_boundary,
            tech_gdf=domestic_tech_gdf,
            line_overlay_gdf=empty_gdf,
            polygon_overlay_gdf=empty_gdf,
            combined_anchor_gdf=empty_gdf,
            radius=50,
            id_col="building_id",
        )

        # Check only domestic building IDs are retained
        results = clusters_gdf.sjoin(
            domestic_tech_gdf, how="inner", predicate="contains"
        )["building_id"]

        expected = domestic_tech_gdf["building_id"]
        extra = set(results).difference(set(expected))
        missing = set(expected).difference(set(results))
        assert (
            not missing
        ), f"Some domestic buildings not contained by a cluster. Building IDs: {missing}"
        assert (
            not extra
        ), f"Some non-domestic building footprints are found in the clusters. Building IDs: {extra}"

        # Check no clusters are empty
        results = (
            clusters_gdf.sjoin(domestic_tech_gdf, how="left", predicate="contains")[
                "building_id"
            ]
            .isna()
            .sum()
        )
        assert (
            results == 0
        ), f"{results} clusters do not contain any domestic building footprints"

    def test_clusters_not_overlapping(
        self,
        gdf_mixed_buildings,
        gdf_enclosing_boundary,
        tech_gdf,
        empty_gdf,
    ):
        """Test there are no overlapping clusters."""
        results = generate_gdf_clusters(
            buildings_gdf=gdf_mixed_buildings,
            boundary_gdf=gdf_enclosing_boundary,
            tech_gdf=tech_gdf,
            line_overlay_gdf=empty_gdf,
            polygon_overlay_gdf=empty_gdf,
            combined_anchor_gdf=empty_gdf,
            radius=50,
            id_col="building_id",
        )

        # Rounding accounts for tiny errors caused by earlier rounding
        assert round(results["geometry"].area.sum(), 8) == round(
            results["geometry"].union_all().area, 8
        ), "Clusters may be overlapping"

    def test_cluster_dissolve(
        self,
        gdf_mixed_buildings,
        gdf_enclosing_boundary,
        tech_gdf,
        empty_gdf,
    ):
        """Test dissolve of neighbouring clusters worked."""
        results = generate_gdf_clusters(
            buildings_gdf=gdf_mixed_buildings,
            boundary_gdf=gdf_enclosing_boundary,
            tech_gdf=tech_gdf,
            line_overlay_gdf=empty_gdf,
            polygon_overlay_gdf=empty_gdf,
            combined_anchor_gdf=empty_gdf,
            radius=50,
            id_col="building_id",
        )

        resulting_tech_types = results["assigned_tech"].to_list()

        n_expected_clusters = 9
        expected_tech_types = [
            "Communal solution",
            "Communal solution",
            "Communal solution",
            "District heat network",
            "Individual solution",
            "Individual solution",
            "Individual solution",
            "Networked heat pump",
            "Networked heat pump",
        ]

        assert len(results) == n_expected_clusters, f"Unexpected number of clusters"
        assert sorted(resulting_tech_types) == sorted(
            expected_tech_types
        ), "Test tech types do not match expected tech types"


class TestExtendEdgesGdf:
    @pytest.fixture(scope="class")
    def gdf_polygons_across_boundary(self):
        """Generate a geodataframe of polygons, some of which are inside; outside; or crossing a boundary."""
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
        """Generate a shapely.Polygon of a boundary to be crossed."""
        return Polygon(
            [
                (399978.0, 399880.0),
                (400100.0, 399880.0),
                (400100.0, 400030.0),
                (399978.0, 400030.0),
            ]
        )

    def test_voronoi_contain_single_polygons(
        self, gdf_mixed_buildings, gdf_enclosing_boundary
    ):
        """Test one Voronoi polygon contains one building footprint."""
        # Access the shapely polygon of the enclosing boundary
        boundary = gdf_enclosing_boundary.geometry.iloc[0]
        cells_gdf = extend_edges_gdf(gdf=gdf_mixed_buildings, boundary=boundary)

        results = cells_gdf[["geometry"]].sjoin(
            gdf_mixed_buildings, how="inner", predicate="contains"
        )["building_id"]
        expected = gdf_mixed_buildings["building_id"]
        missing = set(expected).difference(set(results))

        # All buildings should match to a cell
        assert set(results) == set(
            expected
        ), f"Some buildings are not contained by a Voronoi polygon. Building IDs: {missing}"

        cells_gdf["geometry"] = cells_gdf["geometry"].make_valid().normalize()
        # Buildings and cells should have a 1-1 mapping
        assert (
            cells_gdf["building_id"].duplicated().sum() == 0
        ), f"{cells_gdf['building_id'].duplicated().sum()} building IDs joined to multiple cells"
        assert (
            cells_gdf["geometry"].duplicated().sum() == 0
        ), f"{cells_gdf['geometry'].duplicated().sum()} cells joined to multiple buildings"

    def test_polygons_outside_boundary(
        self, gdf_polygons_across_boundary, geometry_boundary_crossed
    ):
        """Test clustering with polygons outside or crossing the given boundary."""
        cells_gdf = extend_edges_gdf(
            gdf=gdf_polygons_across_boundary, boundary=geometry_boundary_crossed
        )
        results = cells_gdf["building_id"]
        expected = gdf_polygons_across_boundary[
            gdf_polygons_across_boundary["within_boundary"]
        ]["building_id"]

        # TODO update when buildings crossing boundaries has been handled differently
        assert set(results) == set(
            expected
        ), "Polygons outside or crossing boundaries are not handled correctly"

    @pytest.fixture(scope="class")
    def gdf_far_apart_polygons(self):
        """Generate a geodataframe with polygons 100m apart."""
        # Define two buildings with 100m distance between them
        # Building A at X=400000, Building B at X=400100
        return gpd.GeoDataFrame(
            {
                "building_id": ["B01", "B02"],
                "geometry": [
                    Polygon(
                        [
                            (400000, 399995),
                            (400010, 399995),
                            (400010, 400005),
                            (400000, 400005),
                        ]
                    ),
                    Polygon(
                        [
                            (400100, 399995),
                            (400110, 399995),
                            (400110, 400005),
                            (400100, 400005),
                        ]
                    ),
                ],
            },
            crs="EPSG:27700",
        )

    @pytest.fixture(scope="class")
    def geometry_far_apart_boundary(self, gdf_far_apart_polygons):
        """Generate a boundary geometry with a 30m buffer."""
        combined_polygon = gdf_far_apart_polygons.geometry.union_all()
        return shapely.buffer(combined_polygon, distance=30).convex_hull

    def test_clip_voronoi_to_buffer(
        self, gdf_far_apart_polygons, geometry_far_apart_boundary
    ):
        """Test Voronoi polygons are clipped to the desired buffer."""
        cells_gdf = extend_edges_gdf(
            gdf=gdf_far_apart_polygons, boundary=geometry_far_apart_boundary, buffer=20
        )
        results = cells_gdf.area.sum()
        # join_style=2 creates mitred corners of polygon buffers (i.e. sharp rather than rounded corners)
        # Same join_style as used in cluster._clip_gdf_voronoi_cells_polygon_buffer
        expected = gdf_far_apart_polygons.buffer(20, join_style=2).area.sum()

        assert results == expected, "Voronoi not clipped to buffer correctly"
