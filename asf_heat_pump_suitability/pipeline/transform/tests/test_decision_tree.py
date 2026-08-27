"""
Tests for asf_heat_pump_suitability.pipeline.transform.decision_tree.

Run:
pytest asf_heat_pump_suitability/pipeline/transform/tests/test_decision_tree.py
"""

import geopandas as gpd
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Point, Polygon

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.transform import decision_tree

TECH_TYPES = config["constant"]["tech_types"]
COMMUNAL_ORIGIN = config["constant"]["communal_origin"]


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


@pytest.fixture(scope="module")
def tech_gdf():
    """
    Generate UPRN-level decision tree outputs for four buildings:
    - B_FLATS: both UPRNs communal (block of flats)
    - B_FLATS_MIX: one communal UPRN and one networked UPRN
    - B_NET: one networked UPRN
    - B_IND_NET: one individual and one networked UPRN
    """
    rows = [
        ("U01", TECH_TYPES["communal"], "B_FLATS", None),
        ("U02", TECH_TYPES["communal"], "B_FLATS", None),
        ("U03", TECH_TYPES["communal"], "B_FLATS_MIX", None),
        ("U04", TECH_TYPES["networked"], "B_FLATS_MIX", 10.0),
        ("U05", TECH_TYPES["networked"], "B_NET", 12.0),
        ("U06", TECH_TYPES["individual"], "B_IND_NET", 45.0),
        ("U07", TECH_TYPES["networked"], "B_IND_NET", 5.0),
    ]
    return gpd.GeoDataFrame(
        {
            "UPRN": [row[0] for row in rows],
            "assigned_tech": [row[1] for row in rows],
            "ID": [row[2] for row in rows],
            "max_contiguous_outdoor_space_area_m2": [row[3] for row in rows],
        },
        geometry=[Point(400000 + i, 400000) for i in range(len(rows))],
        crs="EPSG:27700",
    )


class TestIdentifyDfBuildingMostSuitableTech:
    """Tests for `identify_df_building_most_suitable_tech`."""

    @pytest.fixture(scope="class")
    def solutions_df(self, tech_gdf):
        """Run the function once and index the result by building ID."""
        return decision_tree.identify_df_building_most_suitable_tech(
            tech_gdf, id_col="ID"
        ).set_index("ID")

    def test_all_communal_building_gets_block_of_flats_origin(self, solutions_df):
        """A building where every UPRN is communal resolves to communal with block-of-flats origin."""
        assert (
            solutions_df.loc["B_FLATS", "assigned_tech"] == TECH_TYPES["communal"]
        ), "a building whose UPRNs are all communal must resolve to communal"
        assert (
            solutions_df.loc["B_FLATS", "communal_origin"]
            == COMMUNAL_ORIGIN["block_of_flats"]
        ), "a communal building resolved by the decision tree must carry the block-of-flats origin"

    def test_mixed_building_containing_communal_gets_block_of_flats_origin(
        self, solutions_df
    ):
        """A building mixing communal and networked UPRNs resolves to communal with block-of-flats origin."""
        assert (
            solutions_df.loc["B_FLATS_MIX", "assigned_tech"] == TECH_TYPES["communal"]
        ), "communal must take precedence in a building mixing communal and networked UPRNs"
        assert (
            solutions_df.loc["B_FLATS_MIX", "communal_origin"]
            == COMMUNAL_ORIGIN["block_of_flats"]
        ), "a building resolved communal because it contains flats must carry the block-of-flats origin"

    def test_non_communal_buildings_have_null_origin(self, solutions_df):
        """Buildings not resolved to communal have a null communal origin."""
        for building in ["B_NET", "B_IND_NET"]:
            assert pd.isna(
                solutions_df.loc[building, "communal_origin"]
            ), f"non-communal building {building} must have a null communal origin"


class TestAssignDfUniqueSolution:
    """Tests for `assign_df_unique_solution`."""

    @pytest.fixture(scope="class")
    def result_df(self):
        """Resolve one building containing communal and one without."""
        solutions_per_footprint_df = pl.DataFrame(
            {
                "ID": ["B1", "B2"],
                "assigned_tech": [
                    [TECH_TYPES["communal"], TECH_TYPES["networked"]],
                    [TECH_TYPES["networked"], TECH_TYPES["individual"]],
                ],
                "median_contiguous_outdoor_space_area_m2": [None, 10.0],
            }
        )
        return decision_tree.assign_df_unique_solution(solutions_per_footprint_df)

    def test_communal_set_gets_block_of_flats_origin(self, result_df):
        """A solution set containing communal resolves to communal with block-of-flats origin."""
        row = result_df.filter(pl.col("ID") == "B1")
        assert (
            row["assigned_tech"].item() == TECH_TYPES["communal"]
        ), "a solution set containing communal must resolve to communal"
        assert (
            row["communal_origin"].item() == COMMUNAL_ORIGIN["block_of_flats"]
        ), "a set containing communal comes from a block of flats so must carry that origin"

    def test_non_communal_set_gets_null_origin(self, result_df):
        """A solution set without communal has a null communal origin."""
        row = result_df.filter(pl.col("ID") == "B2")
        assert (
            row["communal_origin"].item() is None
        ), "a building not resolved to communal must have a null communal origin"
