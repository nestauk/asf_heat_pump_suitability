"""
Unit tests for functions in compute_contextual_features.py
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.run.compute_contextual_features import (
    create_json_contextual_features_metadata,
    extend_gdf_logic_trace,
)

TECH_TYPES = config["constant"]["tech_types"]
COMMUNAL_ORIGIN = config["constant"]["communal_origin"]


@pytest.fixture(scope="module")
def clusters_gdf():
    """
    Generate one cluster per logic-trace branch, including: each communal origin with and without
    DHN potential, the non-communal techs with and without DHN potential, and one
    unexpected combination.
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
        # Points stand in for the cluster polygons: this test does not check geometry
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


class TestCreateJsonContextualFeaturesMetadata:
    """Tests for `create_json_contextual_features_metadata`."""

    @pytest.fixture(scope="class")
    def geojson(self):
        """Run the function once on two clusters and one anchor load, in EPSG:4326.

        `reassigning_anchor_load_ids` cells are numpy arrays, as geoparquet returns list columns.
        """
        clusters_gdf = gpd.GeoDataFrame(
            {
                "cluster_id": ["C01", "C02"],
                "assigned_tech": [TECH_TYPES["communal"], TECH_TYPES["individual"]],
                "communal_origin": [COMMUNAL_ORIGIN["anchor_proximity"], None],
                "reassigning_anchor_load_ids": [
                    np.array(["A1", "A2"], dtype=object),
                    None,
                ],
            },
            geometry=[Point(-4.14, 50.37), Point(-4.13, 50.37)],
            crs="EPSG:4326",
        )
        anchors_gdf = gpd.GeoDataFrame(
            {"anchor_load_id": ["A1"]}, geometry=[Point(-4.15, 50.37)], crs="EPSG:4326"
        )
        return create_json_contextual_features_metadata(
            clusters_with_contextual_features_gdf=clusters_gdf,
            local_authorities="Plymouth",
            release_date="20260901",
            optional_data_layers={"anchor_loads": anchors_gdf},
        )

    @pytest.fixture(scope="class")
    def features_by_layer(self, geojson):
        """Group feature properties by layer."""
        grouped = {}
        for feature in geojson["features"]:
            grouped.setdefault(feature["properties"]["layer"], []).append(
                feature["properties"]
            )
        return grouped

    def test_cluster_features_carry_anchor_id_list(self, geojson, features_by_layer):
        """Anchor-load origin clusters carry a JSON array of anchor IDs; other clusters carry null; the metadata describes it."""
        clusters = {
            props["cluster_id"]: props
            for props in features_by_layer["clusters_with_contextual_features"]
        }
        assert clusters["C01"]["reassigning_anchor_load_ids"] == [
            "A1",
            "A2",
        ], "an anchor-load origin cluster must serialise its anchor IDs as a JSON array"
        assert (
            clusters["C02"]["reassigning_anchor_load_ids"] is None
        ), "a cluster with no linked anchor loads must serialise a null reassigning_anchor_load_ids"
        assert set(clusters["C01"]) == {
            "cluster_id",
            "assigned_tech",
            "communal_origin",
            "reassigning_anchor_load_ids",
            "layer",
        }, "existing cluster properties must pass through unchanged"
        assert (
            "reassigning_anchor_load_ids"
            in geojson["metadata"]["Variable names and descriptions"]
        ), "the geojson metadata must describe the new `reassigning_anchor_load_ids` property"

    def test_anchor_features_carry_anchor_id(self, geojson, features_by_layer):
        """Anchor-load features keep their anchor_load_id and are tagged with the layer name."""
        assert features_by_layer["anchor_loads"] == [
            {"anchor_load_id": "A1", "layer": "anchor_loads"}
        ], "each anchor-load feature must carry its anchor_load_id and layer tag"
        anchor_geometries = [
            feature["geometry"]
            for feature in geojson["features"]
            if feature["properties"]["layer"] == "anchor_loads"
        ]
        assert anchor_geometries == [
            {"type": "Point", "coordinates": [-4.15, 50.37]}
        ], "each anchor-load feature must carry its geometry"
        assert (
            "anchor_load_id" in geojson["metadata"]["Variable names and descriptions"]
        ), "the geojson metadata must describe the new `anchor_load_id` property"
