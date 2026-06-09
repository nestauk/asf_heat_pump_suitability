import pytest
import geopandas as gpd
from shapely.geometry import Polygon, Point
from asf_heat_pump_suitability.pipeline.transform.uprns import (
    map_dict_uprns_to_building_id,
)


class TestMapDictUPRNsBuildingToID:
    """Tests for `map_dict_uprns_to_building_id`.`"""

    @pytest.fixture(scope="class")
    def buildings_gdf(self):
        """Create test building footprints."""
        buildings_data = [
            {
                "building_id": "B1",
                "type": "Square",
                "geometry": Polygon(
                    [
                        (500000, 200000),
                        (500010, 200000),
                        (500010, 200010),
                        (500000, 200010),
                    ]
                ),
            },
            {
                "building_id": "B2",
                "type": "Long Terrace",
                # Shares the eastern side of B1 (from 500010,200000 to 500010,200010)
                "geometry": Polygon(
                    [
                        (500010, 200000),
                        (500040, 200000),
                        (500040, 200010),
                        (500010, 200010),
                    ]
                ),
            },
            {
                "building_id": "B3",
                "type": "L-Shape",
                # Placed further north; will NOT intersect with any points
                "geometry": Polygon(
                    [
                        (500000, 200020),
                        (500020, 200020),
                        (500020, 200030),
                        (500010, 200030),
                        (500010, 200040),
                        (500000, 200040),
                    ]
                ),
            },
        ]

        return gpd.GeoDataFrame(buildings_data, crs="EPSG:27700")

    @pytest.fixture()
    def buildings_gdf_wgs84(self):
        """Create test building footprints in WGS84."""
        buildings_data_wgs84 = [
            {
                "building_id": "B1",
                "type": "Square",
                "geometry": Polygon(
                    [
                        (-0.5547295, 51.6898588),
                        (-0.5545849, 51.6898570),
                        (-0.5545820, 51.6899469),
                        (-0.5547266, 51.6899487),
                    ]
                ),
            },
        ]

        return gpd.GeoDataFrame(buildings_data_wgs84, crs="EPSG:4326")

    @pytest.fixture()
    def gdf_one_uprn_inside_building(self):
        """Create test UPRN geometry inside building."""
        uprns_data = [
            {
                "UPRN": "P1",
                "description": "Inside building footprint",
                "geometry": Point(500005, 200005),  # Inside B1
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    @pytest.fixture()
    def gdf_one_uprn_outside_building(self):
        """Create test UPRN geometry just outside building."""
        uprns_data = [
            {
                "UPRN": "P4",
                "description": "Just outside (<5m outside)",
                "geometry": Point(500043, 200005),  # 3m east of B2's eastern wall
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    @pytest.fixture()
    def gdf_one_uprn_far_outside_building(self):
        """Create test UPRN geometry far outside building (>10m)."""
        uprns_data = [
            {
                "UPRN": "P5",
                "description": "Far outside (>10m outside)",
                "geometry": Point(
                    500000, 200060
                ),  # 20m north of B3, far from all others
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    @pytest.fixture()
    def gdf_many_uprn_to_one_building(self):
        """Create multiple test UPRN geometries to join to one building."""
        uprns_data = [
            {
                "UPRN": "P_B3_1",
                "target_building": "B3",
                "description": "Scattered inside the lower-left of the L-shape",
                "geometry": Point(500005, 200025),
            },
            {
                "UPRN": "P_B3_2",
                "target_building": "B3",
                "description": "Scattered inside the lower-right of the L-shape",
                "geometry": Point(500015, 200025),
            },
            {
                "UPRN": "P_B3_3",
                "target_building": "B3",
                "description": "Scattered inside the upper extension of the L-shape",
                "geometry": Point(500005, 200035),
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    @pytest.fixture()
    def gdf_uprn_intersecting_boundary(self):
        """Create test UPRN geometry intersecting boundary of a building."""
        uprns_data = [
            {
                "UPRN": "P2",
                "description": "On the exterior boundary",
                "geometry": Point(500000, 200005),  # On the western edge of B1
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    @pytest.fixture()
    def gdf_uprn_on_shared_edge(self):
        """Create test UPRN geometry on shared edge of two buildings."""
        uprns_data = [
            {
                "UPRN": "P3",
                "description": "On the shared side of two buildings",
                "geometry": Point(500010, 200005),  # Exactly on the B1/B2 shared line
            },
        ]

        return gpd.GeoDataFrame(uprns_data, crs="EPSG:27700")

    def test_uprn_inside_building(self, buildings_gdf, gdf_one_uprn_inside_building):
        """Test one UPRN maps to the building ID it is located within."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_one_uprn_inside_building,
            id_col="building_id",
            max_distance=1,
        )

        expected = {"P1": "B1"}
        assert results == expected

    def test_diff_crs(self, buildings_gdf_wgs84, gdf_one_uprn_inside_building):
        """Test one UPRN maps to the building ID it is located within when building_gdf and uprns_gdf are in different
        CRS."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf_wgs84,
            uprns_gdf=gdf_one_uprn_inside_building,
            id_col="building_id",
            max_distance=1,
        )

        expected = {"P1": "B1"}
        assert results == expected

    def test_uprn_outside_building(self, buildings_gdf, gdf_one_uprn_outside_building):
        """Test one UPRN maps to the building ID it is located just outside of (<5m)."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_one_uprn_outside_building,
            id_col="building_id",
            max_distance=5,
        )

        expected = {"P4": "B2"}
        assert results == expected

    def test_uprn_no_building_match(
        self, buildings_gdf, gdf_one_uprn_far_outside_building
    ):
        """Test one UPRN does not map to any building due to distance from nearest building (>5m)."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_one_uprn_far_outside_building,
            id_col="building_id",
            max_distance=5,
        )

        expected = dict()
        assert results == expected

    def test_many_to_one_mapping(self, buildings_gdf, gdf_many_uprn_to_one_building):
        """Test many UPRNs map to one building ID."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_many_uprn_to_one_building,
            id_col="building_id",
            max_distance=5,
        )

        expected = {
            "P_B3_1": "B3",
            "P_B3_2": "B3",
            "P_B3_3": "B3",
        }
        assert results == expected

    def test_intersecting_uprn_joined_to_building(
        self, buildings_gdf, gdf_uprn_intersecting_boundary
    ):
        """Test UPRN on building boundary maps to building."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_uprn_intersecting_boundary,
            id_col="building_id",
            max_distance=5,
        )

        expected = {"P2": "B1"}
        assert results == expected

    def test_uprn_joined_to_single_building(
        self, buildings_gdf, gdf_uprn_on_shared_edge
    ):
        """Test UPRN on shared edge maps to only one building."""
        results = map_dict_uprns_to_building_id(
            buildings_gdf=buildings_gdf,
            uprns_gdf=gdf_uprn_on_shared_edge,
            id_col="building_id",
            max_distance=5,
        )

        expected = {"P3": "B1"}
        assert results == expected
