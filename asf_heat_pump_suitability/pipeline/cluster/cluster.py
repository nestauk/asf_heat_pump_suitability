import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import MultiPoint, LineString, Point, box, Polygon, MultiPolygon
import libpysal
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries

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
    # TODO - deal with buildings that cross boundaries
    # Ensure all buildings are within the boundary
    gdf = gdf[gdf.within(boundary)]

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
    ).clip(boundary)

    return gpd.GeoDataFrame(
        gdf.drop(columns=["geometry"])
        .merge(voronoi_gdf, how="inner", on=id_col)
        .drop(columns=["index_right", id_col]),
        geometry="geometry",
        crs=gdf.crs,
    )


def overlay_gdf_physical_barriers(
    voronoi_gdf: gpd.GeoDataFrame,
    tech_gdf: gpd.GeoDataFrame,
    line_overlay_gdf,
    polygon_overlay_gdf,
) -> gpd.GeoDataFrame:
    # Filter to domestic building Voronois only
    voronoi_gdf = voronoi_gdf.sjoin(
        tech_gdf[["geometry"]], how="inner", predicate="contains"
    ).drop(columns="index_right")

    # Remove areas covered by polygons and lines
    cells_gdf = (
        voronoi_gdf.overlay(polygon_overlay_gdf, how="difference")
        .overlay(line_overlay_gdf, how="difference")
        .explode()
    )

    # Remove polygons that no longer intersect with a building
    return cells_gdf.sjoin(
        tech_gdf[["geometry"]], how="inner", predicate="intersects"
    ).drop(columns=["index_right"])


def tranform_gdf_linestring_barriers(grid_squares) -> gpd.GeoDataFrame:
    # Linestrings
    roads_gdf = load_geodata.load_gdf_os_openmap_local_layer(
        layer="road", grid_squares=grid_squares
    )
    railways_gdf = load_geodata.load_gdf_os_openmap_local_layer(
        layer="railway_track", grid_squares=grid_squares
    )

    barrier_road_types = [
        "A Road" "B Road, Collapsed Dual Carriageway",
        "Minor Road, Collapsed Dual Carriageway",
        "Primary Road, Collapsed Dual Carriageway",
        "Motorway",
        "Motorway, Collapsed Dual Carriageway",
        "A Road, Collapsed Dual Carriageway",
    ]

    barrier_roads_gdf = roads_gdf[roads_gdf["CLASSIFICA"].isin(barrier_road_types)]

    line_overlays = [barrier_roads_gdf, railways_gdf]
    line_overlay_gdf = pd.concat([gdf[["geometry"]] for gdf in line_overlays])

    # TODO make more specific for different road types
    # Add buffer assumed to be width of road / railway (3.5m total - 1.75m either side)
    barrier_roads_gdf = [roads_gdf, railways_gdf]
    line_overlay_gdf["geometry"] = line_overlay_gdf.geometry.buffer(1.75)

    return line_overlay_gdf


def transform_gdf_polygon_barriers(grid_squares) -> gpd.GeoDataFrame:
    # Polygons
    forest_gdf = load_geodata.load_gdf_os_openmap_local_layer(
        layer="woodland", grid_squares=grid_squares
    )
    greenspace_gdf = gpd.read_file("path")
    surface_water_gdf = gpd.read_file("path")
    tidal_water_gdf = load_geodata.load_gdf_os_openmap_local_layer(
        layer="tidal_water", grid_squares=grid_squares
    )

    polygon_overlays = [forest_gdf, greenspace_gdf, tidal_water_gdf, surface_water_gdf]

    return pd.concat([gdf[["geometry"]] for gdf in polygon_overlays])


def reassign_gdf_communal_networked(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    shared_tech_gdf = gdf[gdf["tech"].isin(["Networked GSHP", "Communal solutions"])]

    # Create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(shared_tech_gdf)

    # Get component labels
    shared_tech_gdf["components"] = W.component_labels
    gshp_components = shared_tech_gdf[shared_tech_gdf["tech"] == "Networked GSHP"][
        "components"
    ].unique()

    shared_tech_gdf["tech"] = np.where(
        shared_tech_gdf["components"].isin(gshp_components),
        "Networked GSHP",
        shared_tech_gdf["tech"],
    )

    other_tech_gdf = gdf[
        ~gdf["tech"].isin(["Networked GSHP", "Communal solutions"])
    ].reset_index()

    return pd.concat([other_tech_gdf, shared_tech_gdf.reset_index()])


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    # TODO this is a placeholder and likely to change as the script develops
    parser.add_argument(
        "--tech_gdf",
        help="Path to S3 file containing building footprints with their tech types assigned by the decision tree.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--local_authorities",
        help="Run script for either all of Great Britain; Plymouth only {plymouth}; or Plymouth and 4 similar local authorities {plymouth_similar_cities}; or Plymouth and 5 different local authorities {sampling_areas}. Default to all of GB",
        type=str,
        default="GB",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    tech_gdf = gpd.read_file(args.tech_gdf)
    grid_squares = config["constant"]["grid_squares"][args.local_authorities]

    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=config["constant"][args.local_authorities]
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=grid_squares
    )
    line_overlay_gdf = tranform_gdf_linestring_barriers(grid_squares)
    polygon_overlay_gdf = transform_gdf_polygon_barriers(grid_squares)

    gdfs = []

    for boundary in boundary_gdf["geometry"].unique():
        voronoi_gdf = extend_edges_gdf(gdf=buildings_gdf, boundary=boundary)
        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
        )
        gdfs.append(reassign_gdf_communal_networked(cells_gdf))

    clusters_gdf = pd.concat(gdfs)
