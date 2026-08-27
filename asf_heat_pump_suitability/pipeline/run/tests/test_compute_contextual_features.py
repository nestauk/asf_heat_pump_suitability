"""
Unit tests for functions in compute_contextual_features.py
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.run.compute_contextual_features import (
    extend_gdf_logic_trace,
)

TECH_TYPES = config["constant"]["tech_types"]
COMMUNAL_ORIGIN = config["constant"]["communal_origin"]


@pytest.fixture(scope="module")
def clusters_gdf():
    """
    Generate one cluster per logic-trace branch: each communal origin with and without
    DHN potential, the non-communal techs with and without DHN potential, and one
    tech no rule matches.
    """
    rows = [
        # (cluster_id, assigned_tech, communal_origin, in_hn_zone, in_city_centre)
        (
            "C01",
            TECH_TYPES["communal"],
            COMMUNAL_ORIGIN["anchor_proximity"],
            "Yes",
            "No",
        ),
        (
            "C02",
            TECH_TYPES["communal"],
            COMMUNAL_ORIGIN["anchor_proximity"],
            "No",
            "No",
        ),
        ("C03", TECH_TYPES["communal"], COMMUNAL_ORIGIN["block_of_flats"], "No", "Yes"),
        ("C04", TECH_TYPES["communal"], COMMUNAL_ORIGIN["block_of_flats"], "No", "No"),
        ("C05", TECH_TYPES["networked"], None, "Yes", "No"),
        ("C06", TECH_TYPES["networked"], None, "No", "No"),
        ("C07", TECH_TYPES["individual_or_networked"], None, "No", "Yes"),
        ("C08", TECH_TYPES["individual_or_networked"], None, "No", "No"),
        ("C09", TECH_TYPES["individual"], None, "Yes", "No"),
        ("C10", TECH_TYPES["individual"], None, "No", "No"),
        ("C11", "Unexpected combination", None, "No", "No"),
    ]
    return gpd.GeoDataFrame(
        {
            "cluster_id": [row[0] for row in rows],
            "assigned_tech": [row[1] for row in rows],
            "communal_origin": [row[2] for row in rows],
            "in_hn_zone": [row[3] for row in rows],
            "in_city_centre": [row[4] for row in rows],
        },
        geometry=[Point(400000 + i, 400000) for i in range(len(rows))],
        crs="EPSG:27700",
    )


class TestExtendGdfLogicTrace:
    """Tests for `extend_gdf_logic_trace`."""

    @pytest.fixture(scope="class")
    def traces(self, clusters_gdf):
        """Run the function once and index the traces by cluster ID."""
        result_gdf = extend_gdf_logic_trace(clusters_gdf)
        return result_gdf.set_index("cluster_id")["logic_trace"]

    def test_every_cluster_gets_a_trace(self, traces):
        """Every matched cluster gets a trace; only the unmatched tech gets the default."""
        default = "Not accounted for in logic trace."
        matched = traces.drop("C11")
        assert (
            matched != default
        ).all(), "every cluster covered by a rule must get a real trace"
        assert (
            traces["C11"] == default
        ), "a cluster no rule matches must get the default trace"

    def test_communal_traces_keyed_on_origin(self, traces):
        """Communal traces explain only their own origin: anchor clusters mention the
        anchor load and not flats; flats clusters mention flats and not the anchor."""
        for cluster in ["C01", "C02"]:
            assert (
                "anchor load" in traces[cluster]
            ), f"anchor-origin cluster {cluster} must mention the anchor load"
            assert (
                "blocks of flats" not in traces[cluster]
            ), f"anchor-origin cluster {cluster} must not mention blocks of flats"
        for cluster in ["C03", "C04"]:
            assert (
                "blocks of flats" in traces[cluster]
            ), f"flats-origin cluster {cluster} must mention blocks of flats"
            assert (
                "anchor load" not in traces[cluster]
            ), f"flats-origin cluster {cluster} must not mention the anchor load"

    def test_dhn_potential_reflected_in_trace(self, traces):
        """Clusters in a DHN-potential area mention it; others do not."""
        dhn_clusters = ["C01", "C03", "C05", "C07", "C09"]
        no_dhn_clusters = ["C02", "C04", "C06", "C08", "C10"]
        for cluster in dhn_clusters:
            assert (
                "potential" in traces[cluster]
            ), f"cluster {cluster} in a DHN-potential area must mention the potential"
        for cluster in no_dhn_clusters:
            assert (
                "potential" not in traces[cluster]
            ), f"cluster {cluster} outside DHN-potential areas must not mention it"

    def test_no_hedging_sentences(self, traces):
        """No trace hedges about what else the cluster might contain."""
        for cluster, trace in traces.items():
            assert (
                "There might also be" not in trace
            ), f"cluster {cluster} trace must not hedge about blocks of flats"
