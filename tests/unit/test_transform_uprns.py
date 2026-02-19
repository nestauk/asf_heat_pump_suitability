"""Unit tests for asf_heat_pump_suitability.pipeline.transform.uprns."""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point

from asf_heat_pump_suitability.pipeline.transform.uprns import (
    filter_gdf_residential_uprns,
    generate_gdf_uprn_coords,
)


@pytest.fixture()
def small_uprns_df() -> pl.DataFrame:
    """Minimal UPRN DataFrame with BNG coordinates."""
    return pl.DataFrame(
        {
            "UPRN": [1, 2, 3, 4, 5],
            "X_COORDINATE": [245000.0, 245100.0, 245200.0, 245300.0, 245400.0],
            "Y_COORDINATE": [60000.0, 60000.0, 60100.0, 60200.0, 60300.0],
        }
    )


def test_generate_gdf_uprn_coords_returns_geodataframe(small_uprns_df: pl.DataFrame) -> None:
    """generate_gdf_uprn_coords should return a GeoDataFrame with point geometries."""
    gdf = generate_gdf_uprn_coords(small_uprns_df)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 27700
    assert all(gdf.geometry.geom_type == "Point")
    assert len(gdf) == len(small_uprns_df)


def test_generate_gdf_uprn_coords_preserves_columns(small_uprns_df: pl.DataFrame) -> None:
    """generate_gdf_uprn_coords should keep all input columns."""
    gdf = generate_gdf_uprn_coords(small_uprns_df)
    assert "UPRN" in gdf.columns
    assert "X_COORDINATE" in gdf.columns
    assert "Y_COORDINATE" in gdf.columns


def test_generate_gdf_uprn_coords_usecols(small_uprns_df: pl.DataFrame) -> None:
    """generate_gdf_uprn_coords with usecols should include coordinate columns automatically."""
    gdf = generate_gdf_uprn_coords(small_uprns_df, usecols=["UPRN"])
    assert "X_COORDINATE" in gdf.columns
    assert "Y_COORDINATE" in gdf.columns
    assert "UPRN" in gdf.columns


@pytest.fixture()
def uprns_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame of 5 UPRNs (3 inside buildings, 2 outside)."""
    return gpd.GeoDataFrame(
        {"UPRN": [1, 2, 3, 4, 5]},
        geometry=[
            Point(0, 0),  # inside building_a
            Point(0, 0),  # inside building_a (shared coords — flat candidate)
            Point(10, 10),  # inside building_b
            Point(100, 100),  # outside all buildings
            Point(200, 200),  # outside all buildings
        ],
        crs="EPSG:27700",
    )


@pytest.fixture()
def buildings_gdf() -> gpd.GeoDataFrame:
    """Two building footprint polygons."""
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"ID": ["building_a", "building_b"]},
        geometry=[box(-5, -5, 5, 5), box(5, 5, 15, 15)],
        crs="EPSG:27700",
    )


@pytest.fixture()
def non_residential_buildings_gdf(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Second building is non-residential."""
    return buildings_gdf[buildings_gdf["ID"] == "building_b"].copy()


def test_filter_gdf_residential_uprns_removes_non_residential(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """filter_gdf_residential_uprns should exclude UPRNs in non-residential buildings."""
    # Patch EPC loaders to return empty sets so we test the geometric logic only
    import asf_heat_pump_suitability.pipeline.transform.uprns as uprns_mod

    monkeypatch.setattr(uprns_mod, "load_set_valid_epc_uprns", lambda epc_type: set())

    result = filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=buildings_gdf,
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )
    # UPRNs 1 and 2 are in building_a (residential), UPRN 3 is in building_b (non-residential)
    # UPRNs 4 and 5 are outside all buildings
    assert set(result["UPRN"]) == {1, 2}
