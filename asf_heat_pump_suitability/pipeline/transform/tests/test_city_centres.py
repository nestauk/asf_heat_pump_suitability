"""
Tests for asf_heat_pump_suitability.pipeline.transform.city_centres.

Run:
pytest asf_heat_pump_suitability/pipeline/transform/tests/test_city_centres.py
"""

import geopandas as gpd
from shapely.geometry import box

from asf_heat_pump_suitability.pipeline.transform.city_centres import (
    filter_gdf_city_centre_min_area,
)


def test_filter_gdf_city_centre_min_area_merges_touching_areas_before_filter():
    """Two touching 60 m² areas pass together (120 m²); one isolated 60 m² area is removed."""
    city_centres_gdf = gpd.GeoDataFrame(
        {"source_annotation": ["City centre areas"] * 3},
        geometry=[box(0, 0, 10, 6), box(10, 0, 20, 6), box(100, 0, 110, 6)],
        crs="EPSG:27700",
    )

    result = filter_gdf_city_centre_min_area(
        city_centres_gdf=city_centres_gdf, min_area_m2=100
    )

    assert len(result) == 1, "Only the merged pair of touching areas should be kept."
    assert (
        result.geometry.iloc[0].area == 120
    ), "The kept part should be the two touching areas merged into one."
    assert list(result.columns) == [
        "source_annotation",
        "geometry",
    ], "Output should keep only the source annotation and geometry columns."
    assert result.crs == "EPSG:27700", "Output should stay in British National Grid."
