"""Unit tests for asf_heat_pump_suitability.pipeline.transform.outdoor_space."""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point, box

from asf_heat_pump_suitability.pipeline.transform.outdoor_space import (
    deduplicate_df_outdoor_space,
    generate_gdf_building_intersections,
    generate_gdf_outdoor_space,
    sjoin_df_uprn_to_outdoor_space,
)


@pytest.fixture()
def land_parcels_gdf() -> gpd.GeoDataFrame:
    """Two non-overlapping land parcels (10x10m each)."""
    return gpd.GeoDataFrame(
        {"NATIONALCADASTRALREFERENCE": ["A1", "B1"]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs="EPSG:27700",
    )


@pytest.fixture()
def building_footprints_gdf() -> gpd.GeoDataFrame:
    """Building that covers the left half of parcel A1 (5x10m = 50m²)."""
    return gpd.GeoDataFrame(
        {"ID": ["bld_1"]},
        geometry=[box(0, 0, 5, 10)],
        crs="EPSG:27700",
    )


def test_generate_gdf_building_intersections_returns_intersection(
    land_parcels_gdf: gpd.GeoDataFrame,
    building_footprints_gdf: gpd.GeoDataFrame,
) -> None:
    """Building intersection should be the overlap area."""
    result = generate_gdf_building_intersections(land_parcels_gdf, building_footprints_gdf)
    assert len(result) == 1
    assert pytest.approx(result.geometry.area.iloc[0], rel=1e-3) == 50.0


def test_generate_gdf_outdoor_space_calculates_remaining_area(
    land_parcels_gdf: gpd.GeoDataFrame,
    building_footprints_gdf: gpd.GeoDataFrame,
) -> None:
    """Outdoor space should be land area minus building footprint area."""
    intersections = generate_gdf_building_intersections(land_parcels_gdf, building_footprints_gdf)
    result = generate_gdf_outdoor_space(intersections, land_parcels_gdf)

    # Parcel A1 (100m²) minus building (50m²) = 50m² outdoor space
    a1_row = result[result["NATIONALCADASTRALREFERENCE"] == "A1"]
    assert len(a1_row) == 1
    assert pytest.approx(a1_row["max_contiguous_outdoor_space_area_m2"].iloc[0], rel=1e-3) == 50.0


def test_deduplicate_df_outdoor_space_keeps_smallest_total() -> None:
    """When a UPRN matches multiple parcels, keep the one with smallest total area."""
    df = pl.DataFrame(
        {
            "UPRN": [1, 1, 2],
            "NATIONALCADASTRALREFERENCE": ["A1", "B1", "C1"],
            "max_contiguous_outdoor_space_area_m2": [50.0, 30.0, 100.0],
            "total_outdoor_space_area_m2": [50.0, 30.0, 100.0],
        }
    )
    result = deduplicate_df_outdoor_space(df)
    assert len(result) == 2
    uprn_1_row = result.filter(pl.col("UPRN") == 1)
    assert uprn_1_row["NATIONALCADASTRALREFERENCE"][0] == "B1"


def test_sjoin_df_uprn_to_outdoor_space() -> None:
    """UPRNs should be joined to the parcel they fall within."""
    uprns_gdf = gpd.GeoDataFrame(
        {"UPRN": [1, 2], "X_COORDINATE": [2.0, 25.0], "Y_COORDINATE": [5.0, 5.0]},
        geometry=[Point(2, 5), Point(25, 5)],
        crs="EPSG:27700",
    )
    outdoor_space_gdf = gpd.GeoDataFrame(
        {
            "NATIONALCADASTRALREFERENCE": ["A1", "B1"],
            "max_contiguous_outdoor_space_area_m2": [50.0, 100.0],
            "total_outdoor_space_area_m2": [50.0, 100.0],
        },
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs="EPSG:27700",
    )
    result = sjoin_df_uprn_to_outdoor_space(uprns_gdf, outdoor_space_gdf)
    assert isinstance(result, pl.DataFrame)
    assert result.filter(pl.col("UPRN") == 1)["NATIONALCADASTRALREFERENCE"][0] == "A1"
    assert result.filter(pl.col("UPRN") == 2)["NATIONALCADASTRALREFERENCE"][0] == "B1"
