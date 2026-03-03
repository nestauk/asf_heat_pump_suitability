import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import MultiPoint, LineString, Point, box, Polygon, MultiPolygon
from asf_heat_pump_suitability import config

# INPUT = output of decision tree - one row per building with assigned tech type

# OUTPUT = neat polygons around groups of properties with reassignment of communal buildings


def extend_edges_gdf(gdf, boundary, segment_distance=1.0):
    """
    Extends the edges of polygons to create a Voronoi diagram filling the surrounding space.
    Rewritten logic based on fieldmaps/edge-extender.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing polygons.
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary to clip Voronoi polygons to.
        segment_distance (float): Distance (in CRS units) to interval points along edges.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of the extended Voronoi regions.
    """
    # Add an internal unique ID to each building
    id_col = "_internal_unique_id"
    gdf[id_col] = np.arange(len(gdf))

    all_points = []
    all_ids = []

    print("Densifying building polygon edges...")
    # Densify polygon edges with additional points to prepare for Voronoi diagram
    for _, row in gdf.iterrows():
        geom = row.geometry
        id = row[id_col]
        # Deal with any multipolygon buildings
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        for poly in polys:
            # Skip geometries which are not polygons
            if not isinstance(poly, Polygon):
                continue

            # Densify exterior ring of each building to create Voronoi from
            exterior = poly.exterior
            # Calculate the number of points required for densifying
            num_pts = int(np.ceil(exterior.length / segment_distance))
            # Return a list of points at each segment-distance-interval along the exterior edge of the building
            pts = [exterior.interpolate(i * segment_distance) for i in range(num_pts)]
            all_points.extend(pts)
            all_ids.extend([id] * len(pts))

    print(f"Generated {len(all_points)} points.")
    # Create a gdf of all densified points, where one row is one point
    points_gdf = gpd.GeoDataFrame({id_col: all_ids}, geometry=all_points, crs=gdf.crs)

    # Convert to a Multipoint collection for Voronoi
    coords = MultiPoint(points_gdf.geometry.tolist())

    print("Computing Voronoi diagram...")
    # Compute Voronoi polygons up to specified boundary
    voronoi_collection = shapely.voronoi_polygons(coords, extend_to=boundary)

    # Convert to a geodataframe
    voronoi_gdf = gpd.GeoDataFrame(geometry=list(voronoi_collection.geoms), crs=gdf.crs)

    print(
        "Joining Voronois to original building footprints and dissolving per footprint..."
    )
    # Join the points to the Voronoi cells and dissolve to get one polygon per building ID
    voronoi_gdf = (
        voronoi_gdf.sjoin(points_gdf, how="inner", predicate="contains")
        .dissolve(by=id_col)
        .reset_index()
    )
    return gpd.GeoDataFrame(
        gdf.drop(columns=["geometry"])
        .merge(voronoi_gdf, how="inner", on=id_col)
        .drop(columns=["index_right", id_col]),
        geometry="geometry",
        crs=gdf.crs,
    )
