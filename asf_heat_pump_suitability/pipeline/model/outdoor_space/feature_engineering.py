"""
Functions to engineer features at the UPRN level for outdoor space model training.

Features are generated from building footprint geodata and UPRN data, including:
    - Building level features (area, perimeter, convexity, vertex count, plot ratio proxy)
    - UPRN level features (number of UPRNs in building, area per UPRN, perimeter to area ratio, voronoi area, nearest neighbor outdoor space sizes and distances
    - One-hot encoded features for spatial signature types and attachment types
"""

import pandas as pd
import numpy as np
import geopandas as gpd

import shapely
from sklearn.neighbors import NearestNeighbors

from asf_heat_pump_suitability.pipeline.cluster import cluster
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


def _get_gdf_uprns_counts_within_radius(
    gdf: gpd.GeoDataFrame, radius_m: int = 100
) -> pd.DataFrame:
    """
    Find the number of UPRNs within a given radius of UPRN point coordinates.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing UPRN point coordinates. Must contain a 'UPRN' column
        radius_m (int): distance (m) to buffer around each point (default 100m)
    Returns:
        pd.DataFrame: DataFrame containing the number of UPRNs within buffer radius of each UPRN point coordinate

    """
    # Create buffered geometries
    buffers_gdf = gdf[["UPRN", "geometry"]].copy()
    buffers_gdf["geometry"] = buffers_gdf.geometry.buffer(radius_m)

    # Spatial join using 'intersects' to ensure point-in-buffer matching
    joined = buffers_gdf.sjoin(
        gdf[["UPRN", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Group by UPRN_left (the buffer's UPRN) and count matches
    counts = joined.groupby("UPRN_left")["UPRN_right"].count()

    # Subtract 1 to exclude the point itself, clamping at 0
    counts = (counts - 1).clip(lower=0).reset_index()
    counts.columns = ["UPRN", "count"]

    return dict(zip(counts["UPRN"], counts["count"]))


def _calculate_gdf_plot_ratio_proxy(
    buildings_gdf: gpd.GeoDataFrame, buffer_radius: int = 100
) -> gpd.GeoDataFrame:
    """
    Create a circle centred around the building/UPRN centroid with a given radius (default 100m) and
    calculate the ratio of building footprint area contained within the circle to the area of the buffer radius.

    This serves as a proxy for plot ratio, which is a measure of how much of the land around a building is occupied by the building itself.

    E.g. a ratio of 1 means the building footprint takes up the whole buffer circle area,
    and a low ratio means the building footprint takes up little of the circle area, indicating a larger plot of land around the building.

    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprint polygons
        buffer_radius (int): distance (m) around building footprint centroid to calculate plot ratio for (default 100m)

    Returns:
        gpd.GeoDataFrame: building footprint polygons with plot ratio proxy in a new column
    """
    # calculate the area of the buffer circle
    buffer_area = np.pi * (buffer_radius**2)

    buildings_gdf = buildings_gdf.copy()  # Avoid modifying the original GeoDataFrame

    # Preserve original index explicitly in a column before buffering
    buildings_gdf["_orig_idx"] = buildings_gdf.index

    # Create the buffers
    buffers_gdf = gpd.GeoDataFrame(
        buildings_gdf[["_orig_idx"]],
        geometry=buildings_gdf.geometry.centroid.buffer(buffer_radius),
        crs=buildings_gdf.crs,
    )

    # Perform a spatial overlay intersection: Intersects buffer geometries with building geometries
    intersections = gpd.overlay(buffers_gdf, buildings_gdf, how="intersection")

    # Calculate area of clipped intersection geometries
    intersections["clipped_area"] = intersections.geometry.area

    # Group by buffer's original index (_orig_idx_1 is left df, _orig_idx_2 is right df)
    grouped_area = intersections.groupby("_orig_idx_1")["clipped_area"].sum()

    # Map back to main DataFrame and fill missing values
    buildings_gdf["plot_ratio_proxy"] = (
        buildings_gdf["_orig_idx"].map(grouped_area) / buffer_area
    )
    return buildings_gdf.drop(columns=["_orig_idx"])


def _compute_voronoi_area(
    gdf: gpd.GeoDataFrame,
    grid_squares: list[str] | str,
    boundary: shapely.Polygon | shapely.MultiPolygon,
) -> gpd.GeoDataFrame:
    """
    Calculates the area within voronoi polygons formed for each UPRN coordinate, with barriers removed.

    Args:
        gdf (gpd.GeoDataFrame): UPRN point coordinates
        grid_squares (list[str] | str): grid squares to get the barrier features for
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary area to clip the voronoi polygons to

    Returns:
        gpd.GeoDataFrame: original gdf with column for voronoi cell (minus barrier features) area in m2
    """
    boundary_geom = boundary.unary_union

    # Drop UPRNs located at the same coordinates
    voronoi_gdf = gdf.copy().drop_duplicates(subset="geometry")

    # Create polygons around UPRNs
    voronoi_series = voronoi_gdf["geometry"].voronoi_polygons(extend_to=boundary_geom)
    voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_series, crs=gdf.crs)

    # load barriers
    barriers = cluster.load_transform_gdf_polygon_barriers(grid_squares=grid_squares)[
        ["geometry"]
    ].reset_index(drop=True)

    # overlay barriers
    voronoi_gdf = voronoi_gdf.overlay(barriers, how="difference").explode()

    # join back to UPRNs to get just the fragments that contain a UPRN
    voronoi_gdf = voronoi_gdf.sjoin(
        gdf[["UPRN", "geometry"]], how="inner", predicate="contains"
    )

    voronoi_gdf["voronoi_area"] = voronoi_gdf.area

    area_summary = voronoi_gdf.groupby("UPRN")["voronoi_area"].sum().reset_index()
    final_gdf = gdf.merge(area_summary, on="UPRN", how="left")
    final_gdf["voronoi_area"] = final_gdf["voronoi_area"].fillna(0)

    final_gdf = final_gdf.drop(columns=["UPRN"])

    return final_gdf


def engineer_gdf_features(
    uprns_df: pd.DataFrame,
    grid_squares: list[str] | str,
    boundary_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    id_col: str = config["constant"]["id"]["building"],
    outdoor_space_col: str = "max_contiguous_outdoor_space_area_m2",
    radius_m: int = 100,
    nn: int = 5,
) -> gpd.GeoDataFrame:
    """
    Engineer features including:
        - Building level features (area, perimeter, convexity, vertex count, plot ratio proxy)
        - UPRN level features (number of UPRNs in building, area per UPRN, perimeter to area ratio, voronoi area, nearest neighbor outdoor space sizes and distances)
        - One-hot encoded features for spatial signature types and attachment types

    Args:
        uprns_df (pd.DataFrame): UPRN level data
        grid_squares (list[str] | str): grid squares to get the barrier features for
        boundary_gdf (gpd.GeoDataFrame): local authority boundary area
        buildings_gdf (gpd.GeoDataFrame): building footprint polygons
        id_col (str): name of ID column to use for merging UPRN data with building footprint data. Defaults to config["constant"]["id"]["building"]
        outdoor_space_col (str): name of the column in `uprns_df` that contains the known outdoor space size. Defaults "max_contiguous_outdoor_space_area_m2"
        radius_m (int): distance (m) to buffer around each point to count UPRNs within. Defaults to 100m
        nn (int): number of nearest neighbors to find for outdoor space features. Defaults to 5

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with features for model training
    """

    # Drop UPRNs with no building footprint match (i.e. no ID match)
    uprns_df = uprns_df.dropna(subset=[id_col, "X_COORDINATE", "Y_COORDINATE"])

    # Clip building footprints to the local authority boundary
    buildings_gdf = buildings_gdf.clip(boundary_gdf)

    # Calculate building area, perimeter and plot ratio proxy
    buildings_gdf["building_area_m2"] = buildings_gdf.area
    buildings_gdf["building_perimeter_m"] = buildings_gdf.length
    buildings_gdf = _calculate_gdf_plot_ratio_proxy(buildings_gdf=buildings_gdf)

    # Get convex hull and calculate convexity ratio: building area / convex hull area
    buildings_gdf["building_convexity"] = (
        buildings_gdf.geometry.area / (buildings_gdf.geometry.convex_hull.area)
    ).fillna(1.0)

    # Get number of points in a building polygon
    buildings_gdf["building_vertex_count"] = shapely.get_num_coordinates(
        buildings_gdf.geometry.values
    )

    # Count of UPRNs per building footprint
    uprns_df["n_uprns_in_building"] = uprns_df.groupby(id_col)[id_col].transform("size")

    # Creating features_df: with building footprint features and UPRN level features
    features_df = uprns_df.merge(
        buildings_gdf.drop("geometry"),
        on=id_col,
        how="left",  # Drop building geometry to prevent conflict when creating point geometries
    )

    # Create a GeoDataFrame with UPRN point geometries
    features_gdf = gpd.GeoDataFrame(
        features_df,
        geometry=gpd.points_from_xy(
            features_df["X_COORDINATE"],
            features_df["Y_COORDINATE"],
            crs="EPSG:27700",
        ),
    )

    features_gdf["area_per_uprn"] = (
        features_gdf["building_area_m2"] / features_gdf["n_uprns_in_building"]
    )
    features_gdf["perimeter_to_area_ratio"] = (
        features_gdf["building_perimeter_m"] / features_gdf["building_area_m2"]
    )

    features_gdf = _compute_voronoi_area(
        gdf=features_gdf, grid_squares=grid_squares, boundary=boundary_gdf
    )

    # One-hot encoding for spatial signature types
    if "spatial_signature_types" in features_gdf.columns:
        features_gdf["spatial_signature_types"] = (
            features_gdf["spatial_signature_types"]
            .astype(str)
            .str.replace(r"[\[\]\'\"]", "", regex=True)
        )
        spatial_signature_dummies = (
            features_gdf["spatial_signature_types"]
            .str.get_dummies(sep=", ")
            .add_prefix("spatial_signature_")
        )
        features_gdf = pd.concat(
            [
                features_gdf.drop(columns=["spatial_signature_types"]),
                spatial_signature_dummies,
            ],
            axis=1,
        )

    # One-hot encoding for attachment
    if "ATTACHMENT" in features_gdf.columns:
        features_gdf = pd.get_dummies(
            features_gdf, columns=["ATTACHMENT"], prefix="ATTACHMENT", dtype=int
        )

    # Nearest neighbor outdoor space sizes and distances
    features_gdf = _get_gdf_nn_spatial_features(
        gdf=features_gdf,
        n=nn,
        outdoor_space_col=outdoor_space_col,
    )

    features_gdf[f"n_uprns_within_{radius_m}m"] = features_gdf["UPRN"].map(
        _get_gdf_uprns_counts_within_radius(gdf=features_gdf, radius_m=radius_m)
    )

    return features_gdf
