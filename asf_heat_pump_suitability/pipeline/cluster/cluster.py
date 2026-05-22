"""
Functions to generate clusters of building footprints, where one cluster:
- Contains buildings which are assigned the same tech type
- Contains buildings which are not separated by physical environmental barriers

Contains a script to produce clusters from building footprint polygons with assigned tech types. To run:
python asf_heat_pump_suitability/pipeline/cluster/cluster.py

Required args:
--local_authorities to specify which local authority / authorities to run the script for
--save - Set to save output GeoDataFrame to S3.
"""

from typing import Optional, List
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
import libpysal
import warnings
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries

ANCHOR_RADIUS = config["constant"]["anchor_radius"]

ANCHOR_CATEGORIES = [
    "Primary Education",
    "Museum",
    "Library",
    "Further Education",
    "Secondary Education",
    "Fire Station",
    "Sports And Leisure Centre",
    "Hospital",
    "Higher or University Education",
    "Special Needs Education",
    "Medical Care Accommodation",
    "Non State Primary Education",
    "Non State Secondary Education",
    "Art Gallery",
    "Police Station",
    "Hospice",
    "Airport",
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
    id_col: str = "ID",
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
        id_col (str): building ID column. Default "ID".

    Returns:
        gpd.GeoDataFrame: clusters of building footprints with the same assigned technology, one row per cluster
    """
    gdfs = []

    # Create Voronoi polygons and overlay physical barriers for all local authority boundaries
    for boundary in boundary_gdf["geometry"].unique():
        voronoi_gdf = extend_edges_gdf(gdf=buildings_gdf, boundary=boundary)

        # One cell per building
        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=tech_gdf,
            line_overlay_gdf=line_overlay_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
        )
        # TODO No reassignment based on neighbouring cells - TBC if wanted by user testing
        # gdfs.append(reassign_gdf_communal_networked(cells_gdf))
        gdfs.append(cells_gdf)

    # Concatenate all boundary geodataframes together to get a geodataframe of all cells for the whole area of interest
    if len(gdfs) > 1:
        cells_gdf = pd.concat(gdfs)
    else:
        cells_gdf = gdfs[0]

    # TODO move to proper testing when sample test set available
    if len(cells_gdf) != len(tech_gdf):
        n_cells = len(cells_gdf)
        n_buildings = len(tech_gdf)
        warnings.warn(
            f"The number of cells and the number of buildings are different when they should be the same. "
            f"There is a problem with the clustering. N cells: {n_cells}; N buildings: {n_buildings}",
            UserWarning,
        )

    # Tech reassignment for cells within a certain distance of anchor properties
    reassigned_gdf = reassign_gdf_near_anchor_properties(
        tech_gdf=tech_gdf,
        combined_anchor_gdf=combined_anchor_gdf,
        radius=radius,
    )

    cells_gdf["assigned_tech"] = cells_gdf.ID.map(
        reassigned_gdf.set_index(id_col).to_dict()["assigned_tech"]
    )

    # Add "within_{radius}m_from_anchor_load" as a column to cells_gdf
    cells_gdf = cells_gdf.merge(
        reassigned_gdf[[id_col, f"within_{radius}m_from_anchor_load"]],
        on=id_col,
        how="left",
    )

    # Creating the clusters
    clusters_gdf = (
        cells_gdf.dissolve(by="assigned_tech")
        .explode()
        .reset_index()[
            ["assigned_tech", "geometry", f"within_{radius}m_from_anchor_load"]
        ]
    )

    # Create an ID for each geometry that starts with the tech code and ends with a unique number
    # e.g. COM_1, COM_2, etc.
    clusters_gdf["cluster_id"] = clusters_gdf.groupby("assigned_tech").cumcount()

    clusters_gdf["cluster_id"] = (
        clusters_gdf["assigned_tech"].map(TECH_CODES)
        + "_"
        + (clusters_gdf["cluster_id"] + 1).astype(str)
    )

    # TODO move to testing when sample set available
    if round(clusters_gdf["geometry"].area.sum(), 3) > round(
        clusters_gdf["geometry"].union_all().area, 3
    ):
        warnings.warn(
            "Sum of all cluster areas is greater than the area of the cluster union. "
            "This indicates cluster polygons are overlapping.",
            UserWarning,
        )

    # TODO move to testing when sample set available
    joined_gdf = sjoin_gdf_buildings_to_clusters(
        tech_gdf=tech_gdf, clusters_gdf=clusters_gdf
    )
    n_missing = joined_gdf["cluster_id"].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"There is a problem with the clustering. {n_missing} buildings have not been assigned to a cluster.",
            UserWarning,
        )

    return clusters_gdf


def sjoin_gdf_buildings_to_clusters(
    tech_gdf: gpd.GeoDataFrame, clusters_gdf: gpd.GeoDataFrame
):
    """
    Join buildings to their corresponding cluster.

    Args:
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.
        buildings_gdf (gpd.GeoDataFrame): clusters of building footprints with the same assigned technology, one row per cluster.

    Returns:
        gpd.GeoDataFrame: building footprints with assigned tech type and cluster ID
    """
    buffered_tech_gdf = tech_gdf.copy()
    # Reduce the size of the building footprints slightly so they can be completely contained within the cluster cells
    buffered_tech_gdf["geometry"] = buffered_tech_gdf["geometry"].buffer(-0.1)
    return buffered_tech_gdf.sjoin(clusters_gdf, how="left", predicate="within")


def extend_edges_gdf(
    gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
    spacing: float = 1.0,
    buffer: float = 20.0,
) -> gpd.GeoDataFrame:
    """
    Creates Voronoi polygons around a set of input polygons by interpolating additional points along polygon edges
    to extend Voronoi polygons from.
    Rewritten logic based on fieldmaps/edge-extender.

    Args:
        gdf (gpd.GeoDataFrame): polygons to create Voronoi polygons around.
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary to clip Voronoi polygons to.
        spacing (float): Distance in metres to space interpolating points along polygon edges. Default 1.
        buffer (float): buffer (in metres) around polygons in `gdf` to clip the Voronoi cells to. Default 20.

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
    # Compute Voronoi polygons up to specified boundary, create one Voronoi cell per point
    voronoi_collection = shapely.voronoi_polygons(coords, extend_to=boundary)

    # Convert to a geodataframe
    voronoi_gdf = gpd.GeoDataFrame(geometry=list(voronoi_collection.geoms), crs=gdf.crs)

    print(
        "Joining Voronois to original building footprints and dissolving per footprint..."
    )
    # Join the original building points with IDs to the Voronoi cells and dissolve to get one polygon per internal building ID
    voronoi_gdf = (
        # TODO because of the 'contains' predicate, we lose a small number of Voronoi cells which are missing a tiny edge.
        # This is handled later by retaining the building footprint of these buildings instead.
        # However it could be improved by retaining the max intersection.
        voronoi_gdf.sjoin(points_gdf, how="inner", predicate="contains")
        .dissolve(by=id_col)
        .reset_index()
    ).clip(boundary)

    # Clip Voronoi cells to a max buffer
    print("Clip Voronoi cells to maximum buffer...")
    clipped_voronoi_gdf = _clip_gdf_voronoi_cells_polygon_buffer(
        polygon_gdf=gdf, voronoi_gdf=voronoi_gdf, buffer=buffer, id_col=id_col
    )

    # Return Voronoi cell geometries per building with original building ID
    return gpd.GeoDataFrame(
        gdf.drop(columns=["geometry"])
        .merge(clipped_voronoi_gdf, how="inner", on=id_col)
        .drop(columns=["index_right", id_col]),
        geometry="geometry",
        crs=gdf.crs,
    )


def _clip_gdf_voronoi_cells_polygon_buffer(
    polygon_gdf: gpd.GeoDataFrame,
    voronoi_gdf: gpd.GeoDataFrame,
    buffer: float,
    id_col: str,
) -> gpd.GeoDataFrame:
    """
    Clip Voronoi cells to a specified buffer distance around the original polygons in `polygon_gdf`.

    Args:
        polygon_gdf (gpd.GeoDataFrame): polygons to create Voronoi polygons around.
        voronoi_gdf (gpd.GeoDataFrame): Voronoi cells created around the polygons in `polygon_gdf`.
        buffer (float): buffer (in metres) around polygons in `polygon_gdf` to clip the Voronoi cells to. Default 20.
        id_col (str): name of unique ID column in `polygon_gdf`.

    Returns:
        gpd.GeoDataFrame: clipped Voronoi cells
    """
    # Create a buffered polygon for all polygons
    buffered_gdf = polygon_gdf[[id_col, "geometry"]].copy()
    buffered_gdf["geometry"] = buffered_gdf.geometry.buffer(
        # Use mitre join_style which reduces densification of corner vertices when buffering
        buffer,
        join_style=2,
        mitre_limit=2,
        # Simplify with a tolerance of 1mm to remove vertices which are extremely close together
    ).simplify(0.001)

    # Clip Voronoi cells to the defined buffer area if they are larger than it by calculating intersections
    clipped_gdf = voronoi_gdf.overlay(buffered_gdf, how="intersection")
    # Filter clipped cells (intersections) to ensure each polygon's Voronoi cell is clipped only by its own buffer
    return (
        clipped_gdf[clipped_gdf[f"{id_col}_1"] == clipped_gdf[f"{id_col}_2"]]
        .drop(columns=[f"{id_col}_2"])
        .rename(columns={f"{id_col}_1": id_col})
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
    building footprint becoming joined to multiple cell fragments. This handles the fragments by retaining and unioning
    all the fragments which intersect with the original polygon.

    Also retains the building footprint geometry for any domestic buildings which no longer have a Voronoi cell (due to
    overlay operation).

    Args:
        cells_gdf (gpd.GeoDataFrame): resulting Voronoi cells around domestic building footprints after barriers overlaid.
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.

    Returns:
        gpd.GeoDataFrame: domestic building cells with overlapping physical barriers removed and cell fragments handled
    """
    # Add a temporary ID for each building and cell fragment
    building_id_col = "_internal_building_id"
    cell_id_col = "_internal_cell_fragment_id"

    tech_gdf = tech_gdf.assign(**{building_id_col: np.arange(len(tech_gdf))})
    cells_gdf = cells_gdf.assign(**{cell_id_col: np.arange(len(cells_gdf))})

    # Keep only cell fragments that intersect with a building and label with the ID of the building.
    # We use intersection overlay and get the intersection area between each fragment and building, retaining only the
    # pairing with the largest intersection per cell fragment. This handles cases where one fragment joins to multiple
    # buildings to prevent the final set of cells from containing any overlapping geometries.
    intersections_gdf = gpd.overlay(
        cells_gdf[[cell_id_col, "geometry"]],
        tech_gdf[[building_id_col, "geometry"]],
        how="intersection",
    )
    intersections_gdf["area"] = intersections_gdf.geometry.area
    intersections_gdf["max_intersection"] = intersections_gdf.groupby(cell_id_col)[
        "area"
    ].transform("max")
    intersections_gdf = intersections_gdf[
        intersections_gdf["area"] == intersections_gdf["max_intersection"]
    ].copy()

    # Map building IDs to best intersecting cell fragments
    cells_gdf = cells_gdf.merge(
        intersections_gdf[[cell_id_col, building_id_col]],
        on=cell_id_col,
        how="inner",
    )

    # Clean fragments to avoid bleeding geometries creating neighbour 'swallowing' effects during dissolve.
    # e.g. building A swallows building B's cell due to microscopic overlaps in a cell fragment.
    pure_fragments_gdf = gpd.overlay(
        cells_gdf[[building_id_col, "geometry"]],
        tech_gdf[["geometry"]],
        how="difference",
    )

    # Dissolve building footprints and their cell fragments together
    cols = [building_id_col, "geometry"]
    union_gdf = pd.concat(
        [pure_fragments_gdf[cols], tech_gdf[cols]], ignore_index=True
    ).dissolve(by=building_id_col)

    # Join unionised cells back to original buildings to retain building assets
    cells_gdf = (
        # Drop building geometries before merging as we already have them included in the dissolve above
        tech_gdf.drop(columns="geometry")
        .merge(union_gdf, how="left", on=building_id_col)
        .set_geometry("geometry", crs=cells_gdf.crs)
        .drop(columns=[building_id_col])
    )

    return cells_gdf[~cells_gdf["geometry"].is_empty]


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
    roads_gdf = load_geodata.load_gdf_os_openroad(grid_squares=grid_squares)

    railways_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="railway_track", grid_squares=grid_squares
    )

    barrier_road_types = ["A Road", "B Road", "Motorway", "Minor Road"]

    barrier_roads_gdf = roads_gdf[roads_gdf["function"].isin(barrier_road_types)]

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
    tech_gdf[f"within_{radius}m_from_anchor_load"] = np.where(
        (tech_gdf["distance_m"]).notna(), True, False
    )
    tech_gdf = tech_gdf.drop("distance_m", axis=1)
    return tech_gdf


def append_gdf_heat_network_zone_layer(
    clusters_gdf: gpd.GeoDataFrame, hn_zones_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Append DESNZ heat network zones to clusters geodataframe, where `assigned_tech` is 'DESNZ_HNZ'. DESNZ heat network
    zones are also assigned unique cluster IDs by Local Authority.

    Args:
        clusters_gdf (gpd.GeoDataFrame): clusters with `assigned_tech`, `cluster_id`, and `geometry` columns.
        hn_zones_gdf (gpd.GeoDataFrame): DESNZ heat network zones with geometries.

    Returns:
        gpd.GeoDataFrame: cluster and DESNZ heat network zone geometries
    """
    if len(hn_zones_gdf) > 0:
        id_col = [col for col in hn_zones_gdf.columns if "ID" in col][0]
        hn_zones_gdf = hn_zones_gdf.rename(columns={id_col: "cluster_id"}).assign(
            assigned_tech="DESNZ_HNZ",
            cluster_id=lambda df: "DESNZ_HNZ_"
            + df["cluster_id"].astype(str).str.replace("-", "_"),
        )[["assigned_tech", "geometry", "cluster_id"]]

        return pd.concat([clusters_gdf, hn_zones_gdf], ignore_index=True)
    else:
        return clusters_gdf


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

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
    local_authorities = args.local_authorities
    tolerance_m = config["constant"]["clustering"]["tolerance_m"]
    list_las = config["constant"][local_authorities]["la_names"]

    tech_gdf = (
        gpd.read_parquet(
            config["output"]["dataset"]["buildings_most_suitable_tech"].format(
                local_authorities=local_authorities
            )
        )
        .set_geometry("geometry")
        .to_crs(config["constant"]["target_crs"])
    )
    grid_squares = config["constant"][local_authorities]["grid_squares"]

    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=config["constant"][local_authorities]["la_names"]
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

    # Add heat network zones to clusters_gdf, if they exist
    hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(
        local_authority=local_authorities
    )
    clusters_gdf = append_gdf_heat_network_zone_layer(
        clusters_gdf=clusters_gdf, hn_zones_gdf=hn_zones_gdf
    )

    if args.save:
        # Simplify geometry for file size using tolerance
        clusters_gdf["geometry"] = clusters_gdf["geometry"].simplify(
            tolerance=tolerance_m
        )
        save_utils.save_to_s3(
            clusters_gdf,
            config["output"]["dataset"]["tech_clusters"].format(
                local_authorities=local_authorities,
                tolerance_m=tolerance_m,
            ),
        )
