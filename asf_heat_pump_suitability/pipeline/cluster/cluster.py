"""
Functions to generate clusters of building footprints, where one cluster:
- Contains buildings which are assigned the same tech type
- Contains buildings which are not separated by physical environmental barriers

Contains a script to produce clusters from building footprint polygons with assigned tech types. To run:
python asf_heat_pump_suitability/pipeline/cluster/cluster.py

Required args:
--local_authorities to specify which local authority / authorities to run the script for
--save - Set to save the clusters and anchor loads GeoDataFrames to S3.

Set --release_date to specify the YYYYMMDD dated release directory to read inputs from and
save outputs to. Defaults to running the pipeline using today's date. Multi-day runs
should pass the same --release_date to every stage.
"""

from typing import Optional, List
import argparse
import hashlib
import geopandas as gpd
import pandas as pd
import numpy as np
import polars as pl
import shapely
from shapely.geometry import MultiPoint, Polygon, MultiPolygon, Point
from asf_heat_pump_suitability.pipeline.transform import local_authority
import libpysal
import warnings
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import manifest_utils, save_utils
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries

ANCHOR_RADIUS = config["constant"]["anchor_radius"]

# 6 bytes = 12 hex characters: short enough for the geojson, far beyond GB anchor counts
ANCHOR_ID_DIGEST_BYTES = 6

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

COMMUNAL_ORIGIN = config["constant"]["communal_origin"]

ANCHOR_REASSIGNMENT_MAPPING = {NETWORKED: COMMUNAL}


def generate_gdf_clusters(
    buildings_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    tech_gdf: gpd.GeoDataFrame,
    polygon_overlay_gdf: gpd.GeoDataFrame,
    combined_anchor_gdf: gpd.GeoDataFrame,
    radius: float,
    local_authorities_slug: str,
    id_col: str = "ID",
) -> gpd.GeoDataFrame:
    """
    Generate clusters of building footprints, where one cluster:
    - Contains buildings which are assigned the same tech type, and for communal buildings the same `communal_origin`
    - Contains buildings which are not separated by physical environmental barriers
    - Buildings within a given radius of an anchor are assigned a tech type of 'Communal solutions', if they were assigned N-GSHP by the decision tree

    Args:
        buildings_gdf (gpd.GeoDataFrame): all building footprint polygons for area of interest, including domestic and non-domestic.
        boundary_gdf (gpd.GeoDataFrame): boundaries of Local Authorities to generate clusters for.
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.
        polygon_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)Polygon geometries to separate clusters by.
        combined_anchor_gdf (gpd.GeoDataFrame): combined anchor property lists from important buildings and POI data, with building footprints and `anchor_id`
        radius (float): radius in metres around anchor property within which communal solutions should be assigned
        local_authorities_slug (str): slug of local authority to generate clusters for. Used to create unique cluster IDs.
        id_col (str): building ID column. Default "ID".

    Returns:
        gpd.GeoDataFrame: clusters of building footprints with the same assigned technology, one row per cluster, and additional info including:
             -`f"within_{radius}m_from_anchor_load"` flag flagging whether the cluster is within a certain radius from an anchor load
             -`anchor_ids` listing the anchors that caused buildings in the cluster to be reassigned 'communal', where it applies; null for clusters with none.
    """
    gdfs = []

    # Create Voronoi polygons and overlay physical barriers for all local authority boundaries
    for boundary in boundary_gdf["geometry"].unique():
        bounded_tech_gdf = tech_gdf[tech_gdf.within(boundary)]
        voronoi_gdf = extend_edges_gdf(gdf=buildings_gdf, boundary=boundary)

        # One cell per building
        cells_gdf = overlay_gdf_physical_barriers(
            voronoi_gdf=voronoi_gdf,
            tech_gdf=bounded_tech_gdf,
            polygon_overlay_gdf=polygon_overlay_gdf,
            id_col=id_col,
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

    # Tech reassignment for building footprints within a certain distance of anchor properties
    reassigned_gdf = reassign_gdf_near_anchor_properties(
        tech_gdf=tech_gdf,
        combined_anchor_gdf=combined_anchor_gdf,
        radius=radius,
    )

    reassigned_lookup = reassigned_gdf.set_index(id_col).to_dict()
    cells_gdf["assigned_tech"] = cells_gdf[id_col].map(
        reassigned_lookup["assigned_tech"]
    )
    cells_gdf["communal_origin"] = cells_gdf[id_col].map(
        reassigned_lookup["communal_origin"]
    )

    # Create cluster geometries. Dissolving on (assigned_tech, communal_origin) keeps
    # communal clusters with different origins separate; dropna=False retains the
    # non-communal cells, whose communal_origin is null.
    clusters_gdf = (
        cells_gdf.dissolve(by=["assigned_tech", "communal_origin"], dropna=False)
        .explode()
        .reset_index()[["assigned_tech", "communal_origin", "geometry"]]
    )

    # Create an ID for each geometry that starts with the tech code and ends with a unique number
    # e.g. COM_1, COM_2, etc.
    clusters_gdf["cluster_id"] = clusters_gdf.groupby("assigned_tech").cumcount()

    clusters_gdf["cluster_id"] = (
        clusters_gdf["assigned_tech"].map(TECH_CODES)
        + "_"
        + (clusters_gdf["cluster_id"] + 1).astype(str)
        + "_"
        + local_authorities_slug
    )

    # Join boolean flag and reassigning anchor IDs for each building contained in the cluster back to the cluster to aggregate
    clusters_gdf = clusters_gdf.sjoin(
        reassigned_gdf[
            [f"within_{radius}m_from_anchor_load", "anchor_ids", "geometry"]
        ],
        how="left",
        predicate="contains",
    ).drop(columns="index_right")

    # Sorted unique IDs of the anchors that caused a building in the cluster to be reassigned; null when none did
    anchor_ids = (
        clusters_gdf.dropna(subset="anchor_ids")
        .explode("anchor_ids")
        .groupby("cluster_id")["anchor_ids"]
        .agg(lambda ids: sorted(set(ids)))
    )

    # At this point we have multiple rows of each cluster geometry with one row for every building within the cluster.
    # We need to flatten the cluster geometries to one row per cluster, aggregating the within_anchor_radius boolean flag.
    # Selecting `max` of the boolean will mean any clusters containing one building within the anchor radius will be labelled as within the radius.
    # clusters_gdf = clusters_gdf.dissolve(by="cluster_id", aggfunc="max").reset_index()
    clusters_gdf = (
        clusters_gdf.groupby(by="cluster_id")
        .agg(
            {
                "geometry": "first",
                "assigned_tech": "first",
                "communal_origin": "first",
                f"within_{radius}m_from_anchor_load": "max",
            }
        )
        .join(anchor_ids)
        .reset_index()
        .set_geometry(col="geometry", crs=clusters_gdf.crs)
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
        buildings_gdf=tech_gdf, clusters_gdf=clusters_gdf
    )
    n_missing = joined_gdf["cluster_id"].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"There is a problem with the clustering. {n_missing} buildings have not been assigned to a cluster.",
            UserWarning,
        )

    return clusters_gdf


def sjoin_gdf_buildings_to_clusters(
    buildings_gdf: gpd.GeoDataFrame, clusters_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Join buildings to their corresponding cluster. Building footprints are returned with a -10cm buffer.

    Args:
        buildings_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types.
        clusters_gdf (gpd.GeoDataFrame): clusters of building footprints with the same assigned technology, one row per cluster.

    Returns:
        gpd.GeoDataFrame: building footprints with assigned tech type and cluster ID
    """
    buffered_buildings_gdf = buildings_gdf.copy()
    # Reduce the size of the building footprints slightly so they can be completely contained within the cluster cells
    buffered_buildings_gdf["geometry"] = buffered_buildings_gdf["geometry"].buffer(-0.1)
    return buffered_buildings_gdf.sjoin(clusters_gdf, how="left", predicate="within")


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
    building_id_col = "_internal_building_id"
    gdf[building_id_col] = np.arange(len(gdf))

    all_points = []
    all_building_ids = []

    print("Densifying building polygon edges...")
    # Densify polygon edges with additional points to prepare for Voronoi diagram
    for _, row in gdf.iterrows():
        geom = row.geometry
        # Assign building ID
        building_id = row[building_id_col]
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
            # Add corner vertices of building into list, dropping the last one which is a duplicate of the starting point
            pts.extend(Point(coord) for coord in exterior.coords[:-1])
            all_points.extend(pts)
            all_building_ids.extend([building_id] * len(pts))

    print(f"Generated {len(all_points)} points.")
    # Extract a flat (N, 2) float64 array of all point coordinates
    coords_arr = shapely.get_coordinates(np.array(all_points))

    # ordered=True requires every input coordinate to be unique — GEOS raises GEOSException otherwise.
    # Buildings can (rarely) share corner coordinates (e.g. shared walls), so duplicates can exist.
    # To combat this we jitter duplicates by a 0.1mm x-offset so both buildings keep their seed points.

    # Returns the index of the first occurrence of each unique coordinate pair
    _, first_occ = np.unique(coords_arr, axis=0, return_index=True)
    # Mark every position that is NOT a first occurrence as a duplicate
    is_dup = np.ones(len(coords_arr), dtype=bool)
    is_dup[first_occ] = False
    # Offset each duplicate's x coordinate by a unique multiple of 0.1mm - this should not affect the overall shape of buildings significantly.
    # (progressive multiples ensure jittered points don't collide with each other)
    coords_arr[is_dup, 0] += np.arange(1, is_dup.sum() + 1) * 1e-4

    # Convert to a Multipoint collection for Voronoi
    coords = MultiPoint(coords_arr)

    print("Computing Voronoi diagram...")
    # Compute Voronoi polygons up to specified boundary, create one Voronoi cell per point and retain the original order of points
    voronoi_collection = shapely.voronoi_polygons(
        coords, extend_to=boundary, ordered=True
    )

    # Convert to a geodataframe — with ordered=True the nth cell maps directly to the nth input point
    voronoi_gdf = gpd.GeoDataFrame(
        {
            building_id_col: all_building_ids,
            "geometry": np.array(voronoi_collection.geoms),
        },
        crs=gdf.crs,
    )

    print(
        "Joining Voronois to original building footprints and dissolving per footprint..."
    )

    voronoi_gdf.geometry = voronoi_gdf.geometry.make_valid()
    # Sort building IDs and then get the first index of each unique building ID
    arr = voronoi_gdf.sort_values(building_id_col)
    unique_ids, first_idx = np.unique(arr[building_id_col].values, return_index=True)
    # Split the Voronoi geometries into groups. One group == one building ID
    # first_idx[1:] skips the first element which is always 0. This is because np.split splits the array at the locations of
    # each index. If we included index 0, it would create an empty group at the start.
    geom_groups = np.split(arr.geometry.values, first_idx[1:])
    voronoi_gdf = gpd.GeoDataFrame(
        {building_id_col: unique_ids},
        # For each group of Voronoi polygons, union them into one polygon representing a Voronoi for the whole building
        geometry=[shapely.union_all(g) for g in geom_groups],
        crs=voronoi_gdf.crs,
    ).clip(boundary)

    # Clip Voronoi cells to a max buffer
    print("Clip Voronoi cells to maximum buffer...")
    clipped_voronoi_gdf = _clip_gdf_voronoi_cells_polygon_buffer(
        polygon_gdf=gdf, voronoi_gdf=voronoi_gdf, buffer=buffer, id_col=building_id_col
    )

    # Return Voronoi cell geometries per building with original building ID
    return gpd.GeoDataFrame(
        gdf.drop(columns=["geometry"])
        .merge(clipped_voronoi_gdf, how="inner", on=building_id_col)
        .drop(columns=[building_id_col]),
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

    # Clip each Voronoi cell to its own building's buffer using a 1:1 merge and vectorised intersection.
    # This avoids the full cross-join overhead of overlay by only computing the intersection for matching pairs.
    clipped_gdf = voronoi_gdf.merge(
        buffered_gdf.rename(columns={"geometry": "buffer_geom"}), on=id_col
    )
    clipped_gdf["geometry"] = clipped_gdf["geometry"].intersection(
        clipped_gdf["buffer_geom"]
    )
    return clipped_gdf.set_geometry(col="geometry", crs=voronoi_gdf.crs).drop(
        columns="buffer_geom"
    )


def overlay_gdf_physical_barriers(
    voronoi_gdf: gpd.GeoDataFrame,
    tech_gdf: gpd.GeoDataFrame,
    polygon_overlay_gdf: gpd.GeoDataFrame,
    id_col: str,
) -> gpd.GeoDataFrame:
    """
    Conduct difference overlay of physical barriers onto Voronoi polygons. Physical barriers represent features of the
    environment that enforce separation of clusters of households, e.g. environmental barriers that shared technologies
    would not realistically cross. Physical barriers include non-domestic buildings and the areas around them, and may
    optionally include: roads; rivers; railways; bodies of water; green spaces; woodland.

    Args:
        voronoi_gdf (gpd.GeoDataFrame): Voronoi polygons around building footprints
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types
        polygon_overlay_gdf (gpd.GeoDataFrame): physical barriers with (Multi)Polygon geometries.
        id_col (str): building ID column.

    Returns:
        gpd.GeoDataFrame: domestic building cells with overlapping physical barriers removed
    """
    # # Filter to domestic building Voronois only
    # Add a temporary ID for each Voronoi cell
    cell_id_col = "_internal_cell_fragment_id"

    voronoi_gdf = voronoi_gdf.assign(**{cell_id_col: np.arange(len(voronoi_gdf))})

    # Get the largest intersecting Voronoi cell for each domestic building.
    # This method means that buildings with a Voronoi cell that does not completely contain them (e.g. missing a tiny corner)
    # still retain a Voronoi cell
    intersection_gdf = sjoin_gdf_max_intersection(
        cell_gdf=voronoi_gdf,
        building_gdf=tech_gdf,
        cell_id=cell_id_col,
        building_cols=[id_col, "assigned_tech", "geometry"],
    )
    # Map each building to its corresponding Voronoi ID
    cell_to_building_mapping = intersection_gdf.set_index(cell_id_col)[id_col].to_dict()

    # Use the mapping to label the original Voronoi cells with the correct building ID
    voronoi_gdf["select_id"] = voronoi_gdf[cell_id_col].map(cell_to_building_mapping)
    # Filter to the rows where the building ID matches (i.e. only domestic buildings are retained here)
    domestic_voronoi_gdf = voronoi_gdf[
        voronoi_gdf[id_col] == voronoi_gdf["select_id"]
    ].drop(columns="select_id")

    # Remove areas covered by polygons and lines
    cells_gdf = domestic_voronoi_gdf.overlay(
        polygon_overlay_gdf, how="difference"
    ).explode()

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
    intersections_gdf = sjoin_gdf_max_intersection(
        cell_gdf=cells_gdf,
        building_gdf=tech_gdf,
        cell_id=cell_id_col,
        building_cols=[building_id_col, "geometry"],
    )

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


def sjoin_gdf_max_intersection(
    cell_gdf: gpd.GeoDataFrame,
    building_gdf: gpd.GeoDataFrame,
    cell_id: str,
    cell_cols: list[str] = None,
    building_cols: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Match cells to the buildings they have the greatest intersecting area with.

    Args:
        cell_gdf (gpd.GeoDataFrame): polygons of cells which contain buildings
        building_gdf (gpd.GeoDataFrame): polygons of building footprints
        cell_id (str): name of column containing unique cell ID
        cell_cols (list[str]): list of columns to retain from `cell_gdf`. Default `None` to retain only columns required for
        the operation (`cell_id`, `geometry`).
        building_cols (list[str]): list of columns to retain from `building_gdf`. Default `None` to retain only columns required for
        the operation (`geometry`).

    Returns:
        gpd.GeoDataFrame: cells with the building that has the greatest intersecting area with them. One row per unique cell.
    """
    # Ensure required columns are retained
    if cell_cols:
        cell_cols = set(cell_cols)
        cell_cols.add(cell_id)
        cell_cols.add("geometry")
        cell_cols = list(cell_cols)
    else:
        cell_cols = [cell_id, "geometry"]
    if building_cols:
        building_cols = set(building_cols)
        building_cols.add("geometry")
        building_cols = list(building_cols)
    else:
        building_cols = ["geometry"]

    intersections_gdf = gpd.overlay(
        cell_gdf[cell_cols],
        building_gdf[building_cols],
        how="intersection",
    )
    intersections_gdf["area"] = intersections_gdf.geometry.area
    intersections_gdf["max_intersection"] = intersections_gdf.groupby(cell_id)[
        "area"
    ].transform("max")

    return (
        intersections_gdf[
            intersections_gdf["area"] == intersections_gdf["max_intersection"]
        ]
        .copy()
        .drop(columns=["area", "max_intersection"])
    )


def load_transform_gdf_polygon_barriers(
    grid_squares: Optional[List[str]],
) -> gpd.GeoDataFrame:
    """
    Load physical barriers with (Multi)Polygon geometries for the specified grid squares, these include:
    - Green space
    - Water bodies
    - Tidal boundaries
    - Woodland

    Additionally, load physical barriers with (Multi)LineString geometries - railways, and roads of the following types:
    "A Road", "B Road", "Motorway", "Minor Road"- for the specified grid squares. A buffer is added around each
    geometry to cover the width of the road / railway.

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

    # Linestrings
    roads_gdf = load_geodata.load_gdf_os_openroad(grid_squares=grid_squares)
    barrier_road_types = ["A Road", "B Road", "Motorway", "Minor Road"]
    barrier_roads_gdf = roads_gdf[roads_gdf["function"].isin(barrier_road_types)]

    railways_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="railway_track", grid_squares=grid_squares
    )

    line_overlays = [barrier_roads_gdf, railways_gdf]
    line_overlay_gdf = pd.concat([gdf[["geometry"]] for gdf in line_overlays])

    # TODO make more specific for different road types
    # Add buffer assumed to be width of road / railway (3.5m total - 1.75m either side)
    line_overlay_gdf["geometry"] = line_overlay_gdf.geometry.buffer(1.75)

    overlays = [
        forest_gdf,
        greenspace_gdf,
        tidal_water_gdf,
        surface_water_gdf,
        line_overlay_gdf,
    ]

    return pd.concat([gdf[["geometry"]] for gdf in overlays])


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

    Returns:
        gpd.GeoDataFrame: deduplicated anchor footprints with a geometry-derived `anchor_id` column.
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
    combined_anchor_gdf["geometry"] = combined_anchor_gdf.geometry.normalize()
    combined_anchor_gdf = combined_anchor_gdf.drop_duplicates(["geometry"])
    combined_anchor_gdf["anchor_id"] = generate_series_anchor_ids(
        combined_anchor_gdf["geometry"]
    )
    return combined_anchor_gdf


def generate_series_anchor_ids(geometry: gpd.GeoSeries) -> pd.Series:
    """
    Generate a short hex ID for each anchor footprint from its normalised WKB.

    This function normalises the geometries itself, so the input geometries do not need to be
    normalised first. The ID is stable across runs, releases and local authorities for as long
    as the footprint geometry is unchanged.

    Args:
        geometry (gpd.GeoSeries): anchor footprint geometries.

    Returns:
        pd.Series: anchor IDs, aligned to `geometry`.
    """
    return (
        geometry.normalize()
        .to_wkb()
        .map(
            lambda wkb: hashlib.blake2b(
                wkb, digest_size=ANCHOR_ID_DIGEST_BYTES
            ).hexdigest()
        )
    )


def filter_gdf_anchors_to_save(
    anchor_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    clusters_gdf: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Select the anchor loads to save: every anchor load in the local authority, plus any anchor load outside it
    that a cluster references.

    Reassignment uses anchor loads from whole grid squares, so a building inside the local authority
    can be reassigned by an anchor just over the boundary; keeping those means every ID in
    `anchor_ids` resolves to a saved anchor.

    Args:
        anchor_gdf (gpd.GeoDataFrame): anchor footprints with `anchor_id`.
        boundary_gdf (gpd.GeoDataFrame): local authority boundaries.
        clusters_gdf (pd.DataFrame): clusters with an `anchor_ids` list column.

    Returns:
        gpd.GeoDataFrame: `anchor_id` and `geometry` of the anchors to save.
    """
    referenced_ids = set(clusters_gdf["anchor_ids"].dropna().explode())
    in_la = anchor_gdf.intersects(boundary_gdf.union_all())
    return anchor_gdf[in_la | anchor_gdf["anchor_id"].isin(referenced_ids)][
        ["anchor_id", "geometry"]
    ]


def reassign_gdf_near_anchor_properties(
    tech_gdf: gpd.GeoDataFrame,
    combined_anchor_gdf: gpd.GeoDataFrame,
    radius: float,
) -> gpd.GeoDataFrame:
    """
    Reassign building tech type to communal if within a given radius of an anchor load property, if assigned N-GSHP by the decision tree.
    Buildings getting their technology reassigned due to anchor-proximity get `communal_origin` updated to reflect that,
    and `anchor_ids`: the sorted IDs of every anchor load within the radius, since each one alone would cause the
    reassignment. Buildings already communal keep their original value of `communal_origin`.

    Args:
        tech_gdf (gpd.GeoDataFrame): domestic building footprints with assigned tech types and `communal_origin`.
        combined_anchor_gdf (gpd.GeoDataFrame): combined anchor property lists from important buildings and POI data, with building footprints and `anchor_id`
        radius (float): distance in metres around an anchor, within which buildings will be assigned tech type of 'communal solutions' if they were assigned N-GSHP by the decision tree.
    Returns:
        gpd.GeoDataFrame: one row per building, with `assigned_tech` now reading communal if the building is in
        radius of an anchor property and was assigned N-GSHP by the decision tree, and `anchor_ids` (null unless reassigned).
    """
    # Unique index, so the anchors found for each building map back to exactly one row
    tech_gdf = tech_gdf.reset_index(drop=True)

    # Sorted IDs of every anchor load within the radius of each building; buildings with
    # none are absent from this Series
    anchor_ids_within_radius = (
        tech_gdf[["geometry"]]
        .sjoin(
            combined_anchor_gdf[["anchor_id", "geometry"]],
            predicate="dwithin",
            distance=radius,
        )
        .groupby(level=0)["anchor_id"]
        .agg(lambda ids: sorted(set(ids)))
    )
    near_anchor = tech_gdf.index.isin(anchor_ids_within_radius.index)

    # Flag the buildings reassigned to communal, before reassigning them.
    # Buildings that were already communal keep their decision-tree origin.
    newly_communal = near_anchor & (tech_gdf["assigned_tech"] == NETWORKED)
    tech_gdf["assigned_tech"] = np.where(
        near_anchor,
        tech_gdf["assigned_tech"].replace(ANCHOR_REASSIGNMENT_MAPPING),
        tech_gdf["assigned_tech"],
    )
    tech_gdf["communal_origin"] = np.where(
        newly_communal,
        COMMUNAL_ORIGIN["anchor_proximity"],
        tech_gdf["communal_origin"],
    )
    # Only reassigned buildings keep the anchors' identity
    tech_gdf["anchor_ids"] = anchor_ids_within_radius.reindex(tech_gdf.index).where(
        newly_communal
    )
    # add column with True if near anchor, False if not
    tech_gdf[f"within_{radius}m_from_anchor_load"] = near_anchor
    return tech_gdf


def map_df_uprns_to_clusters(
    uprns_df: pl.DataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    clusters_gdf: gpd.GeoDataFrame,
    building_id: str = "ID",
) -> pl.DataFrame:
    """
    Map UPRNs to clusters they are located within.

    Args:
        uprns_df (pl.DataFrame): UPRNs with building ID column required.
        buildings_gdf (gpd.GeoDataFrame): buildings in area of interest with building ID column.
        clusters_gdf (gpd.GeoDataFrame): clusters with `cluster_id` column required.
        building_id (str): name of building ID column in both `uprns_df` and `buildings_gdf`.

    Returns:
        pl.DataFrame: UPRNs mapped to their clusters.
    """

    building_cluster_mapping = sjoin_gdf_buildings_to_clusters(
        buildings_gdf=buildings_gdf, clusters_gdf=clusters_gdf
    ).dropna(subset="cluster_id")
    building_cluster_mapping = building_cluster_mapping.set_index(building_id)[
        "cluster_id"
    ].to_dict()
    uprns_df = uprns_df.with_columns(
        pl.col(building_id)
        # TODO: investigate why some building IDs are not mapping to clusters.
        .replace_strict(building_cluster_mapping, default=None).alias("cluster_id")
    )

    return uprns_df


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        default=["GB"],
        required=False,
    )

    parser.add_argument(
        "--save",
        help="Set to save the clusters and anchor loads GeoDataFrames to S3.",
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    local_authorities = args.local_authorities

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    release_date = save_utils.get_str_release_date(args.release_date)

    tech_gdf = (
        gpd.read_parquet(
            save_utils.get_str_output_path(
                "buildings_most_suitable_tech",
                release_date=release_date,
                check_exists=True,
                local_authorities=local_authority_dict["url_slug"],
            )
        )
        .set_geometry("geometry")
        .to_crs(config["constant"]["target_crs"])
    )

    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=local_authority_dict[
            "valid_local_authorities"
        ]  # TODO add if statement for running with test dataset (would just take the geometry of the input polygons)
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    # Load and transform physical barriers for clusters
    polygon_overlay_gdf = load_transform_gdf_polygon_barriers(
        local_authority_dict["grid_squares"]
    )

    combined_anchor_gdf = load_transform_anchor_property_gdfs(
        buildings_gdf=buildings_gdf, grid_squares=local_authority_dict["grid_squares"]
    )

    # Generate clusters
    clusters_gdf = generate_gdf_clusters(
        buildings_gdf=buildings_gdf,
        boundary_gdf=boundary_gdf,
        tech_gdf=tech_gdf,
        polygon_overlay_gdf=polygon_overlay_gdf,
        combined_anchor_gdf=combined_anchor_gdf,
        radius=ANCHOR_RADIUS,
        local_authorities_slug=local_authority_dict["url_slug"],
    )

    # Saved so the contextual-features stage draws the same footprints and IDs that the
    # clusters reference
    anchors_gdf = filter_gdf_anchors_to_save(
        anchor_gdf=combined_anchor_gdf,
        boundary_gdf=boundary_gdf,
        clusters_gdf=clusters_gdf,
    )

    if args.save:
        run_params = {
            "local_authorities": args.local_authorities,
            "release_date": release_date,
        }
        for dataset, output_gdf in [
            ("tech_clusters", clusters_gdf),
            ("anchor_loads", anchors_gdf),
        ]:
            output_path = save_utils.get_str_output_path(
                dataset=dataset,
                release_date=release_date,
                local_authorities=local_authority_dict["url_slug"],
            )
            save_utils.save_to_s3(df=output_gdf, path=output_path)
            manifest_utils.generate_and_save_run_manifest_to_s3(
                output_path=output_path,
                stage="cluster",
                local_authority=local_authority_dict["url_slug"],
                row_count=len(output_gdf),
                params=run_params,
            )
