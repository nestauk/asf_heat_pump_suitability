"""Unit tests for asf_heat_pump_suitability.pipeline.impute.property_type."""

import geopandas as gpd
from shapely.geometry import Point

from asf_heat_pump_suitability.pipeline.impute import property_type


def make_uprns_gdf(coords: list[tuple[float, float]], uprns: list[int]) -> gpd.GeoDataFrame:
    """Create a minimal UPRNs GeoDataFrame from coordinate and UPRN lists.

    Args:
        coords: List of (x, y) coordinate tuples.
        uprns: List of UPRN integers (one per coordinate).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with UPRN and geometry columns.
    """
    return gpd.GeoDataFrame(
        {"UPRN": uprns},
        geometry=[Point(x, y) for x, y in coords],
        crs="EPSG:27700",
    )


class TestImputeSetFlatProperties:
    """Tests for impute_set_flat_properties."""

    def test_duplicate_geometry_identified_as_flat(self):
        """UPRNs sharing a geometry are identified as flats."""
        # Two UPRNs at the same point => both are flats
        gdf = make_uprns_gdf(
            coords=[(100.0, 200.0), (100.0, 200.0), (300.0, 400.0)],
            uprns=[1, 2, 3],
        )
        result = property_type.impute_set_flat_properties(uprns_gdf=gdf)
        assert 1 in result
        assert 2 in result
        assert 3 not in result

    def test_unique_geometries_not_flats(self):
        """UPRNs with unique geometries are not identified as flats."""
        gdf = make_uprns_gdf(
            coords=[(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)],
            uprns=[1, 2, 3],
        )
        result = property_type.impute_set_flat_properties(uprns_gdf=gdf)
        assert result == set()

    def test_returns_set(self):
        """Return type is always a set."""
        gdf = make_uprns_gdf(coords=[(0.0, 0.0)], uprns=[99])
        result = property_type.impute_set_flat_properties(uprns_gdf=gdf)
        assert isinstance(result, set)

    def test_all_shared_geometry(self):
        """All UPRNs sharing the same geometry are all flats."""
        gdf = make_uprns_gdf(
            coords=[(50.0, 50.0)] * 5,
            uprns=[10, 20, 30, 40, 50],
        )
        result = property_type.impute_set_flat_properties(uprns_gdf=gdf)
        assert result == {10, 20, 30, 40, 50}

    def test_mixed_shared_and_unique(self):
        """Only UPRNs with shared geometries are returned."""
        gdf = make_uprns_gdf(
            coords=[(10.0, 10.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)],
            uprns=[1, 2, 3, 4],
        )
        result = property_type.impute_set_flat_properties(uprns_gdf=gdf)
        assert result == {1, 2}
        assert 3 not in result
        assert 4 not in result
