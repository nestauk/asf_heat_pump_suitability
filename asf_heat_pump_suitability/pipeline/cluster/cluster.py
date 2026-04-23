"""
Functions to generate clusters of building footprints, where one cluster:
- Contains buildings which are assigned the same tech type
- Contains buildings which are not separated by physical environmental barriers

Contains a script to produce clusters from building footprint polygons with assigned tech types. To run:
python asf_heat_pump_suitability/pipeline/cluster/cluster.py

Required args:
--tech_gdf path to geospatial file containing building footprints labelled with assigned tech type
--local_authorities to specify which local authority / authorities to run the script for
"""

from typing import Optional, List
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
import libpysal
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries

ANCHOR_RADIUS = config["constant"]["anchor_radius"]

ANCHOR_CATEGORIES = [
    "Medical Care Accommodation",
    "Primary Education",
    "Further Education",
    "Secondary Education",
    "Sports And Leisure Centre",
    "Hospital",
    "Higher or University Education",
    "Non State Secondary Education",
    "Airport",
    "Library",
    "Special Needs Education",
    "Non State Primary Education",
    "Art Gallery",
    "Museum",
    "Hospice",
]

TECH_TYPES = config["constant"]["tech_types"]
TECH_CODES = {
    TECH_TYPES[k]: config["constant"]["tech_type_codes"][k] for k in TECH_TYPES
}

NETWORKED = TECH_TYPES["networked"]
COMMUNAL = TECH_TYPES["communal"]

TECH_MAPPING = {NETWORKED: COMMUNAL}


def generate_gdf_clusters(
    buildings_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    tech_gdf: gpd.GeoDataFrame,
    line_overlay_gdf: gpd.GeoDataFrame,
    polygon_overlay_gdf: gpd.GeoDataFrame,
    combined_anchor_gdf: gpd.GeoDataFrame,
    radius: float,
) -> gpd.GeoDataFrame:
    """
    Generate clusters of building footprints, where one cluster:
    - Contains buildings which are assigned the same tech type
    - Contains buildings which are not separated by physical environmental barriers
    - Buildings within a given radius of an anchor are assigned a tech type of 'Communal solutions', if they were assigned N-GSHP by the decision tree

    Args:
        buildings_gdf (gpd.GeoDataFrame): all building footprint polygons for area of interest, including domestic and non-domestic.
        boundary_gdf (gpd.GeoDataFrame): boundaries of Local Authorities to generate clusters for.
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.
        line_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)LineString geometries to separate clusters by.
        polygon_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)Polygon geometries to separate clusters by.
        poi_gdf (gpd.GeoDataFrame): anchor properties dataframe taken from POI data, with point geometries
        important_buildings_gdf (gpd.GeoDataFrame): important building footprint polygons
        radius (float): radius in metres around anchor property within which communal solutions should be assigned

    Returns:
        gpd.GeoDataFrame: clusters of building footprints with the same assigned technology, one row per cluster
    """
    gdfs = []

    # Create Voronoi polygons and overlay physical barriers
    for boundary in boundary_gdf["geometry"].unique():
        voronoi_gdf = extend_edges_gdf(gdf=buildings_gdf, boundary=boundary)

        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
        )
        # gdfs.append(reassign_gdf_communal_networked(cells_gdf))
        gdfs.append(cells_gdf)

    # Generate final clusters
    if len(gdfs) > 1:
        clusters_gdf = pd.concat(gdfs)
    else:
        clusters_gdf = gdfs[0]

    # anchor property tech reassignment
    reassigned_gdf = reassign_gdf_near_anchor_properties(
        tech_gdf=tech_gdf,
        combined_anchor_gdf=combined_anchor_gdf,
        radius=radius,
    )
    clusters_gdf["assigned_tech"] = clusters_gdf.ID.map(
        reassigned_gdf.set_index("ID").to_dict()["assigned_tech"]
    )

    # TODO add ID column for clusters
    clusters_gdf = (
        clusters_gdf.dissolve(by="assigned_tech")
        .explode()
        .reset_index()[["assigned_tech", "geometry"]]
    )

    # create an ID for each geometry that starts with the tech code and ends with a unique number, e.g. COM_1, COM_2, etc.
    clusters_gdf["cluster_id"] = clusters_gdf.groupby("assigned_tech").cumcount()

    clusters_gdf["cluster_id"] = (
        clusters_gdf["assigned_tech"].map(TECH_CODES)
        + "_"
        + (clusters_gdf["cluster_id"] + 1).astype(str)
    )

    return clusters_gdf


def extend_edges_gdf(
    gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
    spacing: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Creates Voronoi polygons around a set of input polygons by interpolating additional points along polygon edges
    to extend Voronoi polygons from.
    Rewritten logic based on fieldmaps/edge-extender.

    Args:
        gdf (gpd.GeoDataFrame): polygons to create Voronoi polygons around.
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary to clip Voronoi polygons to.
        spacing (float): Distance in metres to space interpolating points along polygon edges. Default 1.

    Returns:
        gpd.GeoDataFrame: Voronoi polygons around the original input polygons. One row per original polygon.
    """
    # TODO deal with buildings that cross boundaries
    # Ensure all buildings are within the boundary
    gdf = gdf[gdf.within(boundary)]

    # Add an internal unique ID to each building
    id_col = "_internal_building_id"
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
            num_pts = int(np.ceil(exterior.length / spacing))
            # Return a list of points at each segment-distance-interval along the exterior edge of the building
            pts = [exterior.interpolate(i * spacing) for i in range(num_pts)]
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
    line_overlay_gdf: gpd.GeoDataFrame,
    polygon_overlay_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Conduct difference overlay of physical barriers onto Voronoi polygons. Physical barriers represent features of the
    environment that enforce separation of clusters of households, e.g. environmental barriers that shared technologies
    would not realistically cross. Physical barriers include non-domestic buildings and the areas around them, and may
    optionally include: roads; rivers; railways; bodies of water; green spaces; woodland.

    Args:
        voronoi_gdf (gpd.GeoDataFrame): Voronoi polygons around building footprints
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types
        line_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)LineString geometries
        polygon_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)Polygon geometries.

    Returns:
        gpd.GeoDataFrame: domestic building cells with overlapping physical barriers removed
    """
    # Filter to domestic building Voronois only
    voronoi_gdf = voronoi_gdf.sjoin(
        tech_gdf[["assigned_tech", "geometry"]], how="inner", predicate="contains"
    ).drop(columns="index_right")

    # Remove areas covered by polygons and lines
    cells_gdf = (
        voronoi_gdf.overlay(polygon_overlay_gdf, how="difference")
        .overlay(line_overlay_gdf, how="difference")
        .explode()
    )

    # Deal with buildings that have multiple cell fragments
    # This happens in edge cases where a barrier bisects a Voronoi polygon
    return _handle_gdf_fragmented_cells(cells_gdf=cells_gdf, tech_gdf=tech_gdf)


def _handle_gdf_fragmented_cells(
    cells_gdf: gpd.GeoDataFrame, tech_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Handle fragmented Voronoi cells which are created when a Voronoi cell for a single building footprint is fragmented
    during overlaying the physical barriers.

    E.g. a physical barrier can bisect the Voronoi cell or remove parts of the Voronoi cell. This can result in a single
    building footprint becoming joined to multiple cell fragments. This handles the fragments by retaining the largest
    intersecting fragment for the Voronoi, and discarding the rest.

    Also retains the building footprint geometry for any domestic buildings which no longer have a Voronoi cell (due to
    overlay operation).

    Args:
        cells_gdf (gpd.GeoDataFrame): resulting Voronoi cells around domestic building footprints after barriers overlaid.
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.

    Returns:
        gpd.GeoDataFrame: domestic building cells with overlapping physical barriers removed and cell fragments handled
    """
    # Reduce the polygons by 1cm to avoid unions of touching cells
    # Then union cells with building footprints
    union = pd.concat(
        [cells_gdf["geometry"].buffer(-0.01), tech_gdf["geometry"].buffer(-0.01)]
    ).unary_union

    # Explode union
    union_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(union).explode(index_parts=False),
        crs=config["constant"]["target_crs"],
    )

    # Add 1cm buffer back so that dissolving works later
    union_gdf["geometry"] = union_gdf["geometry"].buffer(0.01)

    # Retain original unionised cell geometry
    union_gdf["unionised_geometry"] = union_gdf["geometry"]

    # Join unionised cells back to original buildings
    cells_gdf = tech_gdf.sjoin(union_gdf, how="left", predicate="intersects").drop(
        columns=["index_right"]
    )

    # If building is missing a Voronoi cell (due to overlay operation), then assign it the building footprint geometry
    # This happens in some edge cases
    cells_gdf["unionised_geometry"] = cells_gdf["unionised_geometry"].fillna(
        cells_gdf["geometry"]
    )

    # Keep the Voronoi cell geometries, filled with building footprints
    return (
        cells_gdf.drop(columns="geometry")
        .rename(columns={"unionised_geometry": "geometry"})
        .set_geometry("geometry", crs=config["constant"]["target_crs"])
    )


def load_tranform_gdf_linestring_barriers(
    grid_squares: Optional[List[str]],
) -> gpd.GeoDataFrame:
    """
    Load physical barriers with (Multi)LineString geometries - major roads and railways - for the specified grid squares. A
    buffer is added around each geometry to cover the width of the road / railway.

    # TODO add road types to docstring

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded.
        Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf

    Returns:
        gpd.GeoDataFrame: physical barriers with (Multi)LineString geometries
    """
    # Linestrings
    roads_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="road", grid_squares=grid_squares
    )
    railways_gdf = load_geodata.load_gdf_os_openmap_layer(
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
    line_overlay_gdf["geometry"] = line_overlay_gdf.geometry.buffer(1.75)

    return line_overlay_gdf


def load_transform_gdf_polygon_barriers(
    grid_squares: Optional[List[str]],
) -> gpd.GeoDataFrame:
    """
    Load physical barriers with (Multi)Polygon geometries for the specified grid squares, these include:
    - Green space
    - Water bodies
    - Tidal boundaries
    - Woodland

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded.
        Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf

    Returns:
        gpd.GeoDataFrame: physical barriers with (Multi)Polygon geometries
    """
    # Polygons
    forest_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="woodland", grid_squares=grid_squares
    )

    greenspace_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="greenspace_site", grid_squares=grid_squares
    )

    surface_water_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="surface_water_area", grid_squares=grid_squares
    )

    tidal_water_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="tidal_water", grid_squares=grid_squares
    )

    polygon_overlays = [forest_gdf, greenspace_gdf, tidal_water_gdf, surface_water_gdf]

    return pd.concat([gdf[["geometry"]] for gdf in polygon_overlays])


def reassign_gdf_communal_networked(
    gdf: gpd.GeoDataFrame, n_gshp: str = NETWORKED, communal: str = COMMUNAL
) -> gpd.GeoDataFrame:
    """
    Reassign technology type of Voronoi polygons labelled with 'Communal solutions' if they are in an island* with Voronoi polygons labelled
    'Networked heat pumps'. Communal solutions polygons in these cases will be relabeled with 'Networked heat pumps'.
    *Island here means polygons which are not separated by physical barriers or empty space.

    Args:
        gdf (gpd.GeoDataFrame): Voronoi polygons or polygon clusters generated from building footprints with `assigned_tech` column. One row per Voronoi polygon / cluster.
        n_gshp (str): name of 'Networked GSHP' solution in `assigned_tech`. Defaults set in config/base.yaml.
        communal (str): name of 'Communal solutions' tech in `assigned_tech`. Defaults set in config/base.yaml.

    Returns:
        gpd.GeoDataFrame: original gdf with 'Communal solutions' replaced with 'Networked heat pumps' if they are in the same
        island.
    """
    # Get gdf of communal tech types
    shared_tech_gdf = gdf[gdf["assigned_tech"].isin([n_gshp, communal])]

    # Create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(shared_tech_gdf)

    # Get component labels
    shared_tech_gdf["components"] = W.component_labels
    gshp_components = shared_tech_gdf[shared_tech_gdf["assigned_tech"] == n_gshp][
        "components"
    ].unique()

    # Replace 'communal' label with Networked GSHP if cluster is in an island with N-GSHP
    shared_tech_gdf["assigned_tech"] = np.where(
        shared_tech_gdf["components"].isin(gshp_components),
        n_gshp,
        shared_tech_gdf["assigned_tech"],
    )

    # Get gdf of remaining tech types to concatenate reassigned data to
    other_tech_gdf = gdf[~gdf["assigned_tech"].isin([n_gshp, communal])].reset_index(
        drop=True
    )

    return pd.concat(
        [
            other_tech_gdf,
            shared_tech_gdf.drop(columns="components").reset_index(drop=True),
        ]
    )


def load_transform_anchor_property_gdfs(
    buildings_gdf: gpd.GeoDataFrame,
    grid_squares: Optional[List[str]],
    anchor_categories=ANCHOR_CATEGORIES,
) -> gpd.GeoDataFrame:
    """
    Load data from POI and important buildings lists, select buildings using anchor property categories, and combine the resultant dataframes

    Args:
        buildings_gdf (gpd.GeoDataFrame): all building footprint polygons for area of interest, including domestic and non-domestic.
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded.
        Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf
        anchor_categories (Optional[List[str]]): list of anchor properties to filter important buildings list by. Defaults to ANCHOR_CATEGORIES
    """
    # select anchors out of important building gdf using anchor_categories list
    # anchor categories list is defined at start of script

    poi_gdf = gpd.read_file(
        config["data"]["processed"]["poi_anchor_properties"]
    ).to_crs(config["constant"]["target_crs"])

    important_building_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="important_building", grid_squares=grid_squares
    )

    important_building_gdf = important_building_gdf[
        important_building_gdf["CLASSIFICA"].isin(anchor_categories)
    ]

    # add building footprint data to POI anchor properties so geometry isn't just a point
    anchors_with_footprint = (
        buildings_gdf.sjoin(poi_gdf, how="inner", predicate="contains")
    ).drop("index_right", axis=1)

    # add POI and important building lists together and remove duplicate buildings. Keep only common columns
    combined_anchor_gdf = pd.concat(
        [anchors_with_footprint, important_building_gdf], join="inner"
    )
    combined_anchor_gdf["geometry"] = combined_anchor_gdf.normalize()
    combined_anchor_gdf = combined_anchor_gdf.drop_duplicates(["geometry"])
    return combined_anchor_gdf


def reassign_gdf_near_anchor_properties(
    tech_gdf: gpd.GeoDataFrame,
    combined_anchor_gdf: gpd.GeoDataFrame,
    radius: float,
) -> gpd.GeoDataFrame:
    """
    Reassign building tech type to communal if within a given radius of an anchor load property, if assigned N-GSHP by the decision tree

    Args:
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.
        combined_anchor_gdf (gpd.GeoDataFrame): combined anchor property lists from important buildings and POI data, with building footprints
        radius (float): distance in metres around an anchor, within which buildings will be assigned tech type of 'communal solutions' if they were assigned N-GSHP by the decision tree.
    Returns:
        gpd.GeoDataFrame: dataframe with `assigned_tech` column now reading communal if property is in radius of an anchor property, and was assigned N-GSHP by the decision tree.
    """
    # Spatial join to find nearest anchor for every building

    tech_gdf = tech_gdf.sjoin_nearest(
        combined_anchor_gdf[["geometry"]],
        how="left",
        max_distance=radius,
        distance_col="distance_m",
    ).drop("index_right", axis=1)

    # distance_m column now reads a number (distance from anchor) for all buildings within radius of anchor, and NaN for all outside of that radius
    # if distance column is not NaN (i.e. building is within the radius of an anchor), reassign tech type according to the map
    tech_gdf["assigned_tech"] = np.where(
        tech_gdf["distance_m"].notna(),
        tech_gdf["assigned_tech"].replace(TECH_MAPPING),
        tech_gdf["assigned_tech"],
    )
    # add column with True if near anchor, False if not
    tech_gdf["within_{radius}m_from_anchor_property"] = np.where(
        (tech_gdf["distance_m"]).notna(), True, False
    )
    tech_gdf = tech_gdf.drop("distance_m", axis=1)
    return tech_gdf


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
        help="Path to S3 geoparquet file containing building footprints with their tech types assigned by the decision tree.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
        type=str,
        default="GB",
        required=False,
    )

    parser.add_argument(
        "--save", help="Set to save output GeoDataFrame to S3.", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tech_gdf = (
        gpd.read_parquet(args.tech_gdf)
        .set_geometry("geometry")
        .to_crs(config["constant"]["target_crs"])
    )
    grid_squares = config["constant"][args.local_authorities]["grid_squares"]

    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=config["constant"][args.local_authorities]["la_names"]
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )

    # Load and transform physical barriers for clusters
    line_overlay_gdf = load_tranform_gdf_linestring_barriers(grid_squares)
    polygon_overlay_gdf = load_transform_gdf_polygon_barriers(grid_squares)

    combined_anchor_gdf = load_transform_anchor_property_gdfs(
        buildings_gdf=buildings_gdf, grid_squares=grid_squares
    )

    # Generate clusters
    clusters_gdf = generate_gdf_clusters(
        buildings_gdf=buildings_gdf,
        boundary_gdf=boundary_gdf,
        tech_gdf=tech_gdf,
        line_overlay_gdf=line_overlay_gdf,
        polygon_overlay_gdf=polygon_overlay_gdf,
        combined_anchor_gdf=combined_anchor_gdf,
        radius=ANCHOR_RADIUS,
    )

    if args.save:
        save_utils.save_to_s3(
            clusters_gdf,
            config["output"]["save_as"]["tech_clusters"].format(
                local_authorities=args.local_authorities
            ),
        )
