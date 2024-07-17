import logging
import geopandas as gpd
import shapely


def transform_gdf_drop_duplicates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drop polygons with the same representative point. This drops both duplicate and nearly identical geometries.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with duplicate polygon geometries dropped
    """
    # TODO: this assumes duplicate polygons will always generate the same representative points. Is this correct?
    gdf["rep_point"] = gdf.representative_point()
    if gdf["rep_point"].nunique() != len(gdf):
        duplicate_count = gdf.duplicated(subset="rep_point").sum()

        logging.info(
            f"Polygons containing same representative point found. "
            f"Dropping {duplicate_count} polygons."
        )

        # Sort values to create replicable duplicate removal process
        gdf = gdf.sort_values(by="geometry").drop_duplicates(
            subset="rep_point", keep="first"
        )

    gdf = gdf.drop("rep_point", axis=1)

    return gdf


def get_polygon_gdf_bounds(gdf: gpd.GeoDataFrame) -> shapely.Polygon:
    """
    Get bounding polygon of GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame

    Returns:
        shapely.Polygon: bounding polygon of GeoDataFrame
    """
    gdf_bounds = gdf["geometry"].total_bounds

    bounds = {
        "minx": gdf_bounds[0],
        "miny": gdf_bounds[1],
        "maxx": gdf_bounds[2],
        "maxy": gdf_bounds[3],
    }

    bbox_polygon = shapely.Polygon(
        [
            [bounds["minx"], bounds["miny"]],
            [bounds["minx"], bounds["maxy"]],
            [bounds["maxx"], bounds["maxy"]],
            [bounds["maxx"], bounds["miny"]],
            [bounds["minx"], bounds["miny"]],
        ]
    )

    return bbox_polygon
