from typing import List
import shapely
import geopandas as gpd


def generate_gdf_building_sections(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
    keep_uprns: List[int] = None,
) -> gpd.GeoDataFrame:
    """ """
    if not isinstance(boundary, shapely.Polygon) and not isinstance(
        boundary, shapely.MultiPolygon
    ):
        raise TypeError("`boundary` argument must be a Polygon or MultiPolygon")

    print("Splitting building footprints with Voronoi polygons...")

    if keep_uprns:
        voronoi_gdf = uprns_gdf[
            uprns_gdf.intersects(boundary) & uprns_gdf["UPRN"].isin(keep_uprns)
        ].copy()
    else:
        voronoi_gdf = uprns_gdf[uprns_gdf.intersects(boundary)].copy()

    voronoi_gdf = (
        voronoi_gdf.dissolve(by=["X_COORDINATE", "Y_COORDINATE"], aggfunc="count")
        .reset_index(drop=True)
        .rename(columns={"UPRN": "n_UPRNs"})
    )
    voronoi_gdf["geometry"] = voronoi_gdf.voronoi_polygons(
        extend_to=boundary
    ).make_valid()
    voronoi_gdf = voronoi_gdf.dropna(subset="geometry")
    voronoi_gdf = gpd.clip(voronoi_gdf, boundary)

    return voronoi_gdf.overlay(buildings_gdf, how="intersection", keep_geom_type=False)
