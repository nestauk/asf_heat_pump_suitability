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
    """Two touching 60 m² areas merge into one 120 m² area and stay. The filter removes one separate 60 m² area."""
    city_centres_gdf = gpd.GeoDataFrame(
        {"source_annotation": ["City centre areas"] * 3},
        geometry=[box(0, 0, 10, 6), box(10, 0, 20, 6), box(100, 0, 110, 6)],
        crs="EPSG:27700",
    )

    # 100 m² is more than one area (60 m²) and less than two touching areas (120 m²)
    result = filter_gdf_city_centre_min_area(
        city_centres_gdf=city_centres_gdf, min_area_m2=100
    )

    assert len(result) == 1, "Only the two touching areas, merged into one, must stay."
    assert (
        result.geometry.iloc[0].area == 120
    ), "The area that stays must be the two touching areas merged (120 m²)."
    assert list(result.columns) == [
        "source_annotation",
        "geometry",
    ], "The output must have only the source_annotation and geometry columns."
    assert (
        result.crs == "EPSG:27700"
    ), "The output must use British National Grid (EPSG:27700)."
