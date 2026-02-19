"""Unit tests for asf_heat_pump_suitability.pipeline.impute.property_type."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from asf_heat_pump_suitability.pipeline.impute.property_type import impute_set_flat_properties


@pytest.fixture()
def uprns_with_shared_coords() -> gpd.GeoDataFrame:
    """GeoDataFrame where UPRNs 2 and 3 share the same location (flats)."""
    return gpd.GeoDataFrame(
        {"UPRN": [1, 2, 3, 4, 5]},
        geometry=[
            Point(0, 0),  # unique
            Point(10, 10),  # shared with UPRN 3
            Point(10, 10),  # shared with UPRN 2
            Point(20, 20),  # unique
            Point(30, 30),  # unique
        ],
        crs="EPSG:27700",
    )


def test_impute_set_flat_properties_identifies_shared_coords(
    uprns_with_shared_coords: gpd.GeoDataFrame,
) -> None:
    """UPRNs sharing coordinates should be classified as flats."""
    flats = impute_set_flat_properties(uprns_with_shared_coords)
    assert flats == {2, 3}


def test_impute_set_flat_properties_no_flats() -> None:
    """All UPRNs with unique coordinates should return an empty set."""
    gdf = gpd.GeoDataFrame(
        {"UPRN": [1, 2, 3]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:27700",
    )
    flats = impute_set_flat_properties(gdf)
    assert flats == set()


def test_impute_set_flat_properties_all_flats() -> None:
    """All UPRNs at the same location should all be classified as flats."""
    gdf = gpd.GeoDataFrame(
        {"UPRN": [10, 20, 30]},
        geometry=[Point(5, 5), Point(5, 5), Point(5, 5)],
        crs="EPSG:27700",
    )
    flats = impute_set_flat_properties(gdf)
    assert flats == {10, 20, 30}
