import pytest
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Polygon
from shapely.affinity import rotate
from asf_heat_pump_suitability.pipeline.cluster.cluster import (
    extend_edges_gdf,
    generate_gdf_clusters,
    overlay_gdf_physical_barriers,
)
from asf_heat_pump_suitability.pipeline.cluster.tests import utils


@pytest.fixture(scope="module")
def gdf_mixed_buildings():
    """
    Generate a geodataframe containing a selection of test building footprints to test clustering across different scenarios:
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
    Generate a boundary geodataframe with a 12m buffer.
    """
    combined_footprints = gdf_mixed_buildings.union_all()
    site_boundary_geom = combined_footprints.buffer(12).convex_hull

    return gpd.GeoDataFrame(
        {"name": ["SITE_BOUNDARY"], "geometry": [site_boundary_geom]},
        crs="EPSG:27700",
    )


@pytest.fixture(scope="module")
def empty_gdf():
    """Create an empty geodataframe."""
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=27700)


@pytest.fixture(scope="module")
def tech_gdf(gdf_mixed_buildings):
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


class TestGenerateGdfClusters:
    """Test the `generate_gdf_clusters` function."""

    def test_all_buildings_assigned_cluster(
        self,
        gdf_mixed_buildings,
        gdf_enclosing_boundary,
        tech_gdf,
        empty_gdf,
    ):
        """Test each domestic building is assigned to a cluster."""
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

        # Check if the uncontained area is effectively zero (e.g., less than 1 square millimeter)
        # This is to account for floating point errors which can cause tiny slivers of building not to be covered by
        # the cluster.
        uncontained_slivers_gdf = gdf_mixed_buildings.overlay(
            clusters_gdf, how="difference", keep_geom_type=False
        )
        uncontained_slivers_gdf["area"] = uncontained_slivers_gdf["geometry"].area
        results = uncontained_slivers_gdf[uncontained_slivers_gdf["area"] > 1e-5]
        assert (
            len(results) == 0
        ), f"Buildings {set(results['building_id'])} are missing significant coverage in the clustering."

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
        non_domestic_gdf = tech_gdf[~tech_gdf["domestic"]]
        results = clusters_gdf.sjoin(
            non_domestic_gdf, how="inner", predicate="intersects"
        )["building_id"]

        assert (
            len(results) == 0
        ), "Some non-domestic building footprints are found in the clusters."

        # Check there are no empty clusters
        empty_clusters_gdf = gdf_mixed_buildings.overlay(
            clusters_gdf, how="intersection", keep_geom_type=False
        ).dissolve(by="cluster_id")
        empty_clusters_gdf["area"] = empty_clusters_gdf["geometry"].area

        # Find smallest building area and subtract small sliver for floating point errors
        smallest_building_area = gdf_mixed_buildings.area.min() - 1e-5

        # Clusters must intersect with at least the area of the smallest building footprint, otherwise they are considered empty
        results = empty_clusters_gdf[
            empty_clusters_gdf["area"] <= smallest_building_area
        ]
        assert (
            len(results) == 0
        ), "There are clusters which do not contain any buildings."

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
        boundary = gdf_enclosing_boundary.geometry.iloc[0]
        cells_gdf = extend_edges_gdf(gdf=gdf_mixed_buildings, boundary=boundary)
        # All buildings should match to a cell
        assert set(cells_gdf["building_id"]) == set(
            gdf_mixed_buildings["building_id"]
        ), f"Unique building IDs in buildings and Voronoi cells do not match"

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
        expected = gdf_far_apart_polygons.buffer(20, join_style=2).area.sum()

        assert results == expected


class TestOverlayGdfPhysicalBarriers:
    """"""

    @pytest.fixture(scope="class")
    def voronoi_gdf(self):
        return gpd.GeoDataFrame()

    @pytest.fixture(scope="class")
    def gdf_line_overlay_mixed_buildings(self):
        """
        Create a geodataframe of line polygons (linestrings converted to polygons) that meet the following criteria:
        1. One road separates the corner of a building off from the rest of the footprint.
        2. One road bisects a building.
        3. The other roads go around and between buildings to separate clusters.
        The buildings affected are separate from those affected by the overlay polygons.
        """
        line_dict = {
            "geometry": [
                Polygon(
                    [
                        (400017.68, 400121.33),
                        (400025.87, 400012.34),
                        (400302.02, 400015.53),
                        (400302.26, 400016.72),
                        (400027.79, 400013.53),
                        (400020.89, 400120.79),
                        (400017.68, 400121.33),
                    ]
                ),
                Polygon(
                    [
                        (400094.55, 400125.95),
                        (400095.66, 399990.09),
                        (400073.84, 399990.27),
                        (400052.29, 399989.81),
                        (400044.42, 399987.56),
                        (400034.56, 399985.46),
                        (400035.32, 399879.01),
                        (400036.30, 399878.96),
                        (400035.16, 399984.38),
                        (400044.23, 399986.65),
                        (400052.20, 399989.26),
                        (400096.17, 399989.56),
                        (400096.69, 400126.13),
                        (400094.55, 400125.95),
                    ]
                ),
                Polygon(
                    [
                        (400025.89, 400012.87),
                        (400026.00, 400000.50),
                        (400051.71, 400002.09),
                        (400052.29, 399989.81),
                        (400055.18, 399989.75),
                        (400055.13, 400013.06),
                        (400051.44, 400012.93),
                        (400051.53, 400003.60),
                        (400027.42, 400002.54),
                        (400027.51, 400012.48),
                        (400025.89, 400012.87),
                    ]
                ),
            ]
        }

        # Dissolve overlapping polygons and explode back to one line per polygon
        return gpd.GeoDataFrame(line_dict, crs="EPSG:27700").dissolve().explode()

    @pytest.fixture(scope="class")
    def gdf_polygon_overlay_mixed_buildings(self):
        """
        Create a geodataframe of overlay polygons that meet the following criteria:
        1. Overlaps with the corner of a building.
        2. Completely overlaps a building.
        3. Bisects a building.
        The buildings affected are separate from those affected by the line polygons.
        """
        polygon_dict = {
            "geometry": [
                Polygon(
                    [
                        (400015.8, 400046.4),
                        (399978.9, 400042.1),
                        (399985.0, 400019.7),
                        (400012.9, 400016.7),
                        (400027.9, 400023.6),
                        (400034.6, 400031.1),
                        (400033.7, 400043.0),
                        (400015.8, 400046.4),
                    ]
                ),
                Polygon(
                    [
                        (399924.5, 400012.3),
                        (399914.9, 400009.9),
                        (399920.2, 399998.9),
                        (399928.7, 399991.5),
                        (399941.9, 399986.9),
                        (399950.8, 399979.8),
                        (399975.7, 399977.0),
                        (399991.4, 399975.2),
                        (400004.4, 399973.2),
                        (400009.9, 399967.0),
                        (400011.3, 399956.7),
                        (400010.6, 399945.2),
                        (400012.9, 399929.9),
                        (400009.5, 399918.4),
                        (400003.5, 399907.0),
                        (399991.0, 399890.9),
                        (400001.7, 399877.6),
                        (400010.9, 399896.7),
                        (400018.1, 399921.6),
                        (400018.4, 399948.9),
                        (400016.5, 399965.8),
                        (400007.2, 399977.3),
                        (399992.6, 399979.4),
                        (399978.7, 399982.5),
                        (399957.4, 399994.5),
                        (399941.0, 399995.2),
                        (399924.5, 400012.3),
                    ]
                ),
                Polygon(
                    [
                        (400062.2, 400007.4),
                        (400062.5, 400004.6),
                        (400065.2, 400004.6),
                        (400066.1, 400007.0),
                        (400062.2, 400007.4),
                    ]
                ),
                Polygon(
                    [
                        (400021.8, 399989.3),
                        (400021.8, 399980.4),
                        (400031.9, 399977.4),
                        (400038.2, 399977.6),
                        (400037.3, 399988.0),
                        (400029.8, 399987.9),
                        (400029.2, 399989.4),
                        (400024.0, 399989.7),
                        (400021.8, 399989.3),
                    ]
                ),
            ]
        }

        return gpd.GeoDataFrame(polygon_dict, crs="EPSG:27700")

    def test_cells_entirely_contain_buildings_after_overlay(
        self,
        voronoi_gdf,
        tech_gdf,
        gdf_line_overlay_mixed_buildings,
        gdf_polygon_overlay_mixed_buildings,
        gdf_mixed_buildings,
    ):
        # TODO requires an edge case input where overlaying barriers remove all and some of the property (including
        #  overlapping and bisecting)
        # TODO edge case where original Voronoi cell doesn't completely cover building
        """Test Voronoi cells entirely contain buildings after overlaying barriers. This tests the function can handle
        edge cases like a barrier completely or partially overlapping a building footprint and thus fragmenting the cell
        so that it no longer encases a full building footprint."""
        domestic_tech_gdf = tech_gdf[tech_gdf["domestic"]]

        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=domestic_tech_gdf,
            line_overlay_gdf=gdf_line_overlay_mixed_buildings,
            polygon_overlay_gdf=gdf_polygon_overlay_mixed_buildings,
            id_col="building_id",
        )

        results = cells_gdf.sjoin(
            gdf_mixed_buildings, how="inner", predicate="contains"
        )["building_id"]
        expected = gdf_mixed_buildings["building_id"]

        assert set(results) == set(expected)
