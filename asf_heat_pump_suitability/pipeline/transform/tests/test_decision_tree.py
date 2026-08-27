"""
Tests for asf_heat_pump_suitability.pipeline.transform.decision_tree.

Run:
pytest asf_heat_pump_suitability/pipeline/transform/tests/test_decision_tree.py
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from asf_heat_pump_suitability.pipeline.transform import decision_tree


@pytest.fixture(scope="module")
def buildings_gdf():
    """Two square building footprints in British National Grid coordinates."""
    return gpd.GeoDataFrame(
        {
            "ID": ["B1", "B2"],
            "geometry": [
                Polygon(
                    [
                        (500000, 200000),
                        (500010, 200000),
                        (500010, 200010),
                        (500000, 200010),
                    ]
                ),
                Polygon(
                    [
                        (500020, 200000),
                        (500030, 200000),
                        (500030, 200010),
                        (500020, 200010),
                    ]
                ),
            ],
        },
        crs="EPSG:27700",
    )


@pytest.fixture(scope="module")
def uprns_gdf():
    """Three UPRNs with the decision tree's input features: two flats in B1
    (one in a city centre / heat network zone), one house in B2."""
    return gpd.GeoDataFrame(
        {
            "UPRN": [1, 2, 3],
            "ID": ["B1", "B1", "B2"],
            "in_block_of_flats": [True, True, False],
            "max_contiguous_outdoor_space_area_m2": [0.0, 0.0, 120.0],
            "in_city_centre_or_hn_zone": [True, True, False],
            "geometry": [
                Point(500005, 200005),
                Point(500006, 200005),
                Point(500025, 200005),
            ],
        },
        crs="EPSG:27700",
    )


class TestIdentifyGdfTupleMostSuitableTechUprnAndBuilding:
    """Tests for `identify_gdf_tuple_most_suitable_tech_uprn_and_building`."""

    def test_returns_uprn_and_building_gdfs_with_assigned_tech(
        self, buildings_gdf, uprns_gdf
    ):
        """Returns one GeoDataFrame per level, each with an assigned_tech column."""
        uprns_tech_gdf, buildings_tech_gdf = (
            decision_tree.identify_gdf_tuple_most_suitable_tech_uprn_and_building(
                buildings_gdf=buildings_gdf,
                id_col="ID",
                uprns_gdf=uprns_gdf,
            )
        )
        assert isinstance(
            uprns_tech_gdf, gpd.GeoDataFrame
        ), "UPRN-level output is not a GeoDataFrame"
        assert isinstance(
            buildings_tech_gdf, gpd.GeoDataFrame
        ), "Building-level output is not a GeoDataFrame"
        assert (
            "assigned_tech" in uprns_tech_gdf.columns
        ), "UPRN-level output is missing the assigned_tech column"
        assert (
            "assigned_tech" in buildings_tech_gdf.columns
        ), "Building-level output is missing the assigned_tech column"
        assert len(uprns_tech_gdf) == len(
            uprns_gdf
        ), "Expected one output row per input UPRN"
        assert (
            len(buildings_tech_gdf) == buildings_gdf["ID"].nunique()
        ), "Expected one output row per input building"

    def test_each_building_gets_a_single_tech(self, buildings_gdf, uprns_gdf):
        """All UPRNs in a building share the building's single assigned tech."""
        uprns_tech_gdf, _ = (
            decision_tree.identify_gdf_tuple_most_suitable_tech_uprn_and_building(
                buildings_gdf=buildings_gdf,
                id_col="ID",
                uprns_gdf=uprns_gdf,
            )
        )
        techs_per_building = uprns_tech_gdf.groupby("ID")["assigned_tech"].nunique()
        assert (techs_per_building == 1).all(), (
            f"Buildings with more than one assigned tech: "
            f"{techs_per_building[techs_per_building > 1].index.tolist()}"
        )
