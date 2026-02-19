"""Integration tests for the UPRN filtering pipeline module.

Tests the public interface of the uprns transform module using small but
realistic synthetic GeoDataFrames.
"""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from asf_heat_pump_suitability.pipeline.transform.uprns import (
    filter_gdf_residential_uprns,
    generate_gdf_uprn_coords,
)


@pytest.fixture()
def synthetic_uprns_df() -> pl.DataFrame:
    """Synthetic UPRN DataFrame with 10 properties."""
    return pl.DataFrame(
        {
            "UPRN": list(range(1, 11)),
            "X_COORDINATE": [float(x) for x in range(0, 100, 10)],
            "Y_COORDINATE": [5.0] * 10,
        }
    )


@pytest.fixture()
def synthetic_buildings_gdf() -> gpd.GeoDataFrame:
    """Five buildings covering the first 5 UPRNs."""
    return gpd.GeoDataFrame(
        {"ID": [f"bld_{i}" for i in range(5)]},
        geometry=[box(x - 2, 0, x + 2, 10) for x in range(0, 50, 10)],
        crs="EPSG:27700",
    )


@pytest.fixture()
def non_residential_buildings_gdf(synthetic_buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """First two buildings are non-residential."""
    return synthetic_buildings_gdf.head(2)


def test_filter_gdf_residential_uprns_count(
    synthetic_uprns_df: pl.DataFrame,
    synthetic_buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residential UPRN filter should return correct count."""
    import asf_heat_pump_suitability.pipeline.transform.uprns as uprns_mod

    monkeypatch.setattr(uprns_mod, "load_set_valid_epc_uprns", lambda epc_type: set())

    uprns_gdf = generate_gdf_uprn_coords(synthetic_uprns_df)
    result = filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=synthetic_buildings_gdf,
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )
    # UPRNs 1-2 are non-residential, UPRNs 3-5 are residential buildings, 6-10 are outside
    assert len(result) == 3
    assert set(result["UPRN"]) == {3, 4, 5}


def test_filter_gdf_residential_uprns_epc_rescue(
    synthetic_uprns_df: pl.DataFrame,
    synthetic_buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UPRNs in domestic EPC register should be kept even if not in a building."""
    import asf_heat_pump_suitability.pipeline.transform.uprns as uprns_mod

    # UPRNs 8 and 9 are outside all buildings but appear in the domestic EPC
    epc_residential = {8, 9}

    def mock_load_epc(epc_type: str) -> set:
        if epc_type == "domestic":
            return epc_residential
        return set()

    monkeypatch.setattr(uprns_mod, "load_set_valid_epc_uprns", mock_load_epc)

    uprns_gdf = generate_gdf_uprn_coords(synthetic_uprns_df)
    result = filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=synthetic_buildings_gdf,
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )
    # Should include EPC-rescued UPRNs
    assert 8 in set(result["UPRN"])
    assert 9 in set(result["UPRN"])


def test_generate_gdf_uprn_coords_schema(synthetic_uprns_df: pl.DataFrame) -> None:
    """Output GeoDataFrame should have expected schema and CRS."""
    gdf = generate_gdf_uprn_coords(synthetic_uprns_df)
    assert gdf.crs.to_epsg() == 27700
    assert "UPRN" in gdf.columns
    assert all(gdf.geometry.geom_type == "Point")
    assert len(gdf) == len(synthetic_uprns_df)
