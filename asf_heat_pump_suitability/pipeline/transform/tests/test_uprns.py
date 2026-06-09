import pytest
import geopandas as gpd
from shapely.geometry import Polygon, Point
from asf_heat_pump_suitability.pipeline.transform.uprns import (
    map_dict_uprns_to_building_id,
)


class TestMapDictUPRNsBuildingToID:
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
    def uprns_gdf(self):
        """Create test UPRN geometries."""
        points_data = [
            {
                "point_id": "P1",
                "description": "Inside building footprint",
                "geometry": Point(500005, 200005),  # Inside B1
            },
            {
                "point_id": "P2",
                "description": "On the exterior boundary",
                "geometry": Point(500000, 200005),  # On the western edge of B1
            },
            {
                "point_id": "P3",
                "description": "On the shared side of two buildings",
                "geometry": Point(500010, 200005),  # Exactly on the B1/B2 shared line
            },
            {
                "point_id": "P4",
                "description": "Just outside (<5m outside)",
                "geometry": Point(500043, 200005),  # 3m east of B2's eastern wall
            },
            {
                "point_id": "P5",
                "description": "Far outside (10+m outside)",
                "geometry": Point(
                    500000, 200060
                ),  # 20m north of B3, far from all others
            },
        ]

        return gpd.GeoDataFrame(points_data, crs="EPSG:27700")

    def test_diff_crs(self):
        pass

    def test_uprn_inside_building(self):
        pass

    def test_uprn_outside_building(self):
        pass

    def test_uprn_no_building_match(self):
        pass

    def test_uprn_joined_to_single_building(self):
        pass
