"""
Functions to engineer features at the UPRN level for outdoor space model training. Features are generated from building footprint geodata and labelled UPRN data.
"""

import pandas as pd
import numpy as np
import geopandas as gpd

import shapely
from sklearn.neighbors import NearestNeighbors

from asf_heat_pump_suitability.pipeline.cluster import cluster
from asf_heat_pump_suitability.pipeline.transform import local_authority
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata
from asf_heat_pump_suitability import config


def _get_gdf_nn_spatial_features(
    gdf: gpd.GeoDataFrame,
    n: int = 5,
    outdoor_space_col: str = "max_contiguous_outdoor_space_area_m2",
) -> gpd.GeoDataFrame:
    """
    Takes a GeoDataFrame of UPRNs, finds the n nearest neighbors that have
    known outdoor space, and extracts their individual outdoor space sizes and distances.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with UPRN point coordinates and a column for known outdoor space size
        n (int): number of nearest neighbors to find (default 5)
        outdoor_space_col (str): name of the column in `gdf` that contains the known outdoor space size (default "max_contiguous_outdoor_space_area_m2")

    Returns:
        gpd.GeoDataFrame: original gdf with columns for n nearest neighbour garden sizes and distances
    """
    # Create a copy of the GeoDataFrame to avoid modifying the original
    gdf = gdf.copy()

    # Extract target coordinates and UPRNs from the GeoDataFrame
    target_coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    target_ids = gdf["UPRN"].to_numpy()

    # Filter to UPRNs with known outdoor space size
    known_gdf = gdf.dropna(subset=[outdoor_space_col]).copy()

    # Round coordinates to 4 decimal places (0.1mm in EPSG:27700) to eliminate micro-duplicates
    coords_round = np.column_stack(
        (
            known_gdf.geometry.centroid.x.round(4),
            known_gdf.geometry.centroid.y.round(4),
        )
    )

    # Drop duplicates based on rounded coordinates and outdoor space size to avoid self-matches
    _, unique_idx = np.unique(
        coords_round, axis=0, return_index=True
    )  # or keep Pandas drop_duplicates
    known_unique = known_gdf.iloc[unique_idx]

    known_coords = np.column_stack(
        (known_unique.geometry.centroid.x, known_unique.geometry.centroid.y)
    )
    known_sizes = known_unique[outdoor_space_col].to_numpy()
    known_ids = known_unique["UPRN"].to_numpy()

    # Deal with unlikely case: no known outdoor space
    if len(known_coords) == 0:
        for j in range(n):
            gdf[f"nn{j+1}_garden_size"] = np.nan
            gdf[f"nn{j+1}_distance_m"] = np.nan
        return gdf

    total_known = len(known_coords)

    # Query up to n + 1 neighbors in case a point matches itself
    query = min(n + 1, total_known)
    # Fit NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=query, algorithm="kd_tree")
    nn.fit(known_coords)
    distances, indices = nn.kneighbors(target_coords)

    # Map neighbor indices back to their IDs and sizes
    retrieved_ids = known_ids[indices]
    retrieved_sizes = known_sizes[indices]

    # Create a 2D boolean mask where True represents a valid neighbor, False represents a self-match
    is_valid_neighbor = retrieved_ids != target_ids[:, None]

    # We sort the mask (~is_valid places False/self-matches at the very end of each row)
    # argsort along axis=1 produces the row indices that shift self-matches out of the top `n` slot
    sort_order = np.argsort(~is_valid_neighbor, axis=1)

    # Re-order 2D matrices
    sorted_sizes = np.take_along_axis(retrieved_sizes, sort_order, axis=1)
    sorted_distances = np.take_along_axis(distances, sort_order, axis=1)
    sorted_valid_mask = np.take_along_axis(is_valid_neighbor, sort_order, axis=1)

    # Mask out invalid positions (if total available valid neighbors < n)
    final_sizes = np.where(sorted_valid_mask[:, :n], sorted_sizes[:, :n], np.nan)
    final_distances = np.where(
        sorted_valid_mask[:, :n], sorted_distances[:, :n], np.nan
    )

    # Create columns in original GeoDataFrame
    size_cols = {f"nn{j+1}_garden_size": final_sizes[:, j] for j in range(n)}
    dist_cols = {f"nn{j+1}_distance_m": final_distances[:, j] for j in range(n)}

    gdf = gdf.assign(**size_cols, **dist_cols)

    return gdf


def _get_gdf_number_uprns_within_radius(
    gdf: gpd.GeoDataFrame, radius_m: int = 100
) -> gpd.GeoDataFrame:
    """
    Find the number of UPRNs within a given radius of UPRN point coordinates.
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing UPRN point coordinates. Must contain a 'UPRN' column
        radius_m (int): distance (m) to buffer around each point (default 100m)
    Returns:
        gpd.GeoDataFrame: number of UPRNs within buffer radius of each UPRN point coordinate

    """

    # Create buffers around the point coordinates
    buffers = gpd.GeoDataFrame(
        {"UPRN": gdf["UPRN"], "geometry": gdf.geometry.buffer(radius_m)}, crs=gdf.crs
    )

    # Create the target points
    points = gpd.GeoDataFrame(
        {"UPRN": gdf["UPRN"], "geometry": gdf.geometry}, crs=gdf.crs
    )

    # find target points inside buffer radius
    joined = gpd.sjoin(buffers, points, how="left", predicate="contains")

    # Group by the correct unique ID and count UPRNs within buffer radius
    uprn_counts = joined.groupby("UPRN").size().reset_index(name="uprns_within_100m")

    return uprn_counts


def _calculate_gdf_plot_ratio_proxy(
    buildings_gdf: gpd.GeoDataFrame, buffer_radius: int = 100
) -> gpd.GeoDataFrame:
    """
    Calculate the ratio of building footprint area contained within a buffer radius to the area of the buffer radius. E.g. a ratio of 1 means the building footprint takes up the whole buffer circle area, and a low area means the building footprint takes up little of the buffer radius
    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprint polygons
        buffer_radius (int): distance (m) around building footprint centroid to calculate plot ratio for (default 100m)

    Returns:
        gpd.GeoDataFrame: building footprint polygons with plot ratio proxy in a new column
    """
    # calculate the area of the buffer circle
    buffer_area = np.pi * (buffer_radius**2)

    # Create the buffers
    buffers = buildings_gdf.geometry.centroid.buffer(buffer_radius)
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, index=buildings_gdf.index)

    # find buildings within buffer circles and keep the buffer geometry
    joined = gpd.sjoin(buffers_gdf, buildings_gdf, how="inner", predicate="intersects")

    left_geoms = joined.geometry  # These are the buffer circles

    # get building polygons that are within the buffer circles
    # the geometry will be of the full building (not just the bit inside the circle)
    # explicitly give them the exact same index as the 'joined' dataframe.
    right_geoms = gpd.GeoSeries(
        buildings_gdf.loc[joined["index_right"], "geometry"].values, index=joined.index
    )

    # now clip building geometries to just what is within the buffer circle areas
    exact_intersections = left_geoms.intersection(right_geoms)

    # Calculate the area of just those clipped pieces
    joined["clipped_area"] = exact_intersections.area

    # Group by the buffer's index and sum the clipped areas
    total_exact_area = joined.groupby(joined.index)["clipped_area"].sum()

    # Calculate the ratio and assign it back to the main dataframe
    buildings_gdf["plot_ratio_proxy"] = total_exact_area / buffer_area

    # Fill NaNs with 0 (in case a point had zero intersecting buildings)
    buildings_gdf["plot_ratio_proxy"] = buildings_gdf["plot_ratio_proxy"].fillna(0)

    return buildings_gdf


def _get_int_count_vertices(geom: shapely.Polygon | shapely.MultiPolygon) -> int:
    """
    Counts the number of vertices in a polygon or mutipolygon
    Args:
        geom (shapely.Polygon | shapely.MultiPolygon): shapely geometry representing building footprint polygon

    Returns:
        int: count of vertices in the polygon or multi polygon
    """
    if geom.geom_type == "Polygon":
        # Count the coordinates forming the outer boundary
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        # Sum the coordinates for all pieces of the multi-polygon
        return sum(len(poly.exterior.coords) for poly in geom.geoms)
    return 0


def _compute_voronoi_area(
    gdf: gpd.GeoDataFrame,
    grid_squares: list[str] | str,
    boundary: shapely.Polygon | shapely.MultiPolygon,
) -> gpd.GeoDataFrame:
    """
    Calculates the area within voronoi polygons formed for each UPRN coordinate, with barrier features removed
    Args:
        gdf (gpd.GeoDataFrame): UPRN point coordinates
        grid_squares (list[str] | str): grid squares to get the barrier features for
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary area to clip the voronoi polygons to

    Returns:
        gpd.GeoDataFrame: original gdf with column for voronoi cell (minus barrier features) area in m2
    """
    boundary_geom = boundary.unary_union

    # Add an internal unique ID to each UPRN
    id_col = "_internal_building_uprn"
    gdf[id_col] = np.arange(len(gdf))

    # Drop UPRNs located at the same coordinates
    voronoi_gdf = gdf.copy().drop_duplicates(subset="geometry")

    # Create polygons around UPRNs
    voronoi_series = voronoi_gdf["geometry"].voronoi_polygons(extend_to=boundary_geom)
    voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_series, crs=gdf.crs)

    # load barriers
    polygon_barriers = cluster.load_transform_gdf_polygon_barriers(
        grid_squares=grid_squares
    )[["geometry"]].reset_index(drop=True)
    linestring_barriers = cluster.load_tranform_gdf_linestring_barriers(
        grid_squares=grid_squares
    )[["geometry"]].reset_index(drop=True)

    # overlay barriers
    voronoi_gdf = (
        voronoi_gdf.overlay(polygon_barriers, how="difference")
        .overlay(linestring_barriers, how="difference")
        .explode()
    )

    # join back to UPRNs to get just the fragments that contian a UPRN
    voronoi_gdf = voronoi_gdf.sjoin(
        gdf[[id_col, "geometry"]], how="inner", predicate="contains"
    )

    voronoi_gdf["voronoi_area"] = voronoi_gdf.area

    area_summary = voronoi_gdf.groupby(id_col)["voronoi_area"].sum().reset_index()
    final_gdf = gdf.merge(area_summary, on=id_col, how="left")
    final_gdf["voronoi_area"] = final_gdf["voronoi_area"].fillna(0)

    final_gdf = final_gdf.drop(columns=[id_col])

    return final_gdf


def engineer_gdf_features(
    local_authorities: str | list[str],
    id_col: str = config["constant"]["id"]["building"],
) -> gpd.GeoDataFrame:
    """
    Engineer set of features for model training at the UPRN level
    Args:
        local_authorities (str | list[str]): Local Authority or Authorities to engineer features for
        id_col (str): name of ID column to use for merging UPRN data with building footprint data. Defaults to config["constant"]["id"]["building"]

    Returns:
        gpd.GeoDataFrame: features for model training
    """

    grid_squares = list(local_authority.get_list_la_grid_squares(local_authorities))

    # get building footprints for LA boundary
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        local_authorities
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )
    buildings_gdf = buildings_gdf.clip(boundary_gdf)

    # add building level features
    buildings_gdf["building_area_m2"] = buildings_gdf.area
    buildings_gdf["building_perimeter_m"] = buildings_gdf.length
    buildings_gdf = _calculate_gdf_plot_ratio_proxy(buildings_gdf=buildings_gdf)

    # get convex hull and calculate convexity ratio
    convex_hull_areas = buildings_gdf.geometry.convex_hull.area
    buildings_gdf["building_convexity"] = (
        buildings_gdf.geometry.area / convex_hull_areas
    )
    buildings_gdf["building_convexity"] = buildings_gdf["building_convexity"].fillna(
        1.0
    )

    # get number of points in a building polygon
    buildings_gdf["building_vertex_count"] = buildings_gdf.geometry.apply(
        _get_int_count_vertices
    )

    # load UPRNs with features
    uprns_df_list = []
    for la in local_authorities:
        la_slug = local_authority.make_str_slug(la)
        uprns_la = pd.read_parquet(
            f"s3://asf-local-heat-planning-tool/outputs/data/{la_slug}/{la_slug}_with_features.parquet"
        )
        uprns_df_list.append(uprns_la)

    uprns_df = pd.concat(uprns_df_list, ignore_index=True)
    uprns_df = uprns_df.dropna(subset=["ID"])

    # add column of UPRNs per building footprint
    uprn_counts = (
        uprns_df.groupby(id_col).size().reset_index(name="n_uprns_in_building")
    )
    uprns_df = uprns_df.merge(uprn_counts, on="ID", how="left")

    # merge building level features onto UPRN data
    df_with_features = pd.merge(uprns_df, buildings_gdf, on="ID", how="left")

    gdf_with_features = gpd.GeoDataFrame(
        df_with_features,
        geometry=gpd.points_from_xy(
            df_with_features["X_COORDINATE"],
            df_with_features["Y_COORDINATE"],
            crs="EPSG:27700",
        ),
    )

    gdf_with_features["area_per_uprn"] = (
        gdf_with_features["building_area_m2"] / gdf_with_features["n_uprns_in_building"]
    )
    gdf_with_features["perimeter_to_area_ratio"] = (
        gdf_with_features["building_perimeter_m"]
        / gdf_with_features["building_area_m2"]
    )

    gdf_with_features = _compute_voronoi_area(
        gdf=gdf_with_features, grid_squares=grid_squares, boundary=boundary_gdf
    )

    # convert spatial signature types to be one column per spatial signature type, with 1 being True and 0 being False
    gdf_with_features["spatial_signature_types"] = (
        gdf_with_features["spatial_signature_types"]
        .astype(str)
        .str.replace(r"[\[\]\'\"]", "", regex=True)
    )
    gdf_with_features = pd.get_dummies(
        gdf_with_features,
        columns=["spatial_signature_types"],
        prefix="spatial_signature",
        dtype=int,
    )
    # as above, for attachment type
    gdf_with_features = pd.get_dummies(
        gdf_with_features, columns=["ATTACHMENT"], prefix="ATTACHMENT", dtype=int
    )
    gdf_with_features = _get_gdf_nn_spatial_features(
        gdf=gdf_with_features,
        n=5,
        outdoor_space_col="max_contiguous_outdoor_space_area_m2",
    )

    uprn_sums = _get_gdf_number_uprns_within_radius(gdf=gdf_with_features)
    gdf_with_features = gdf_with_features.merge(uprn_sums, on="UPRN", how="left")

    return gdf_with_features
