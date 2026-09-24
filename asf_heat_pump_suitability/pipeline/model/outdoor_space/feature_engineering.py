"""
Functions to engineer features at the UPRN level for outdoor space model training. Features are generated from building footprint geodata and labelled UPRN data.
"""

import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.neighbors import NearestNeighbors

from asf_heat_pump_suitability.pipeline.cluster import cluster
from asf_heat_pump_suitability.pipeline.transform import local_authority
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata


def _get_gdf_5nn_spatial_features(gdf, unique_id_col):
    """
    Takes a GeoDataFrame of UPRNs, finds the 5 nearest neighbors that have
    known garden sizes, and extracts their individual sizes and distances.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame of UPRN point coordinates
        unique_id_col (str): name of ID column

    Returns:
        gpd.GeoDataFrame: original gdf with columns for 5 nearest neighbour garden sizes and distances
    """
    # get centroids and IDs of uprns
    all_centroids = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    all_ids = gdf[unique_id_col].values

    # get just the known garden sizes and the centroids of them
    known_gardens = gdf.dropna(subset=["max_contiguous_outdoor_space_area_m2"])

    known_gardens["coord_round"] = (
        known_gardens.geometry.centroid.x.round(4).astype(str)
        + "_"
        + known_gardens.geometry.centroid.y.round(4).astype(str)
    )

    known_gardens_unique = known_gardens.drop_duplicates(
        subset=["coord_round", "max_contiguous_outdoor_space_area_m2"]
    )

    known_centroids = np.column_stack(
        (
            known_gardens_unique.geometry.centroid.x,
            known_gardens_unique.geometry.centroid.y,
        )
    )
    known_sizes = known_gardens_unique["max_contiguous_outdoor_space_area_m2"].values
    known_ids = known_gardens_unique[unique_id_col].values

    # find the 6 nearest garden sizes and the distance to them. We need 6 so that we can drop the self-match later
    nn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    nn.fit(known_centroids)
    distances, indices = nn.kneighbors(all_centroids)

    # Initialize storage lists
    nn1_size, nn2_size, nn3_size, nn4_size, nn5_size = [], [], [], [], []
    nn1_dist, nn2_dist, nn3_dist, nn4_dist, nn5_dist = [], [], [], [], []

    # Loop through the array
    for i in range(len(all_centroids)):
        target_id = all_ids[i]
        row_indices = indices[i]
        row_distances = distances[i]

        # drop the self-match
        neighbor_ids = known_ids[row_indices]
        valid_mask = neighbor_ids != target_id

        valid_sizes = known_sizes[row_indices[valid_mask]]
        valid_distances = row_distances[valid_mask]

        sizes_to_use = list(valid_sizes[:5])
        dists_to_use = list(valid_distances[:5])

        # Pad with NaNs if there are extremely isolated buildings
        while len(sizes_to_use) < 5:
            sizes_to_use.append(np.nan)
            dists_to_use.append(np.nan)

        # Unpack sizes
        nn1_size.append(sizes_to_use[0])
        nn2_size.append(sizes_to_use[1])
        nn3_size.append(sizes_to_use[2])
        nn4_size.append(sizes_to_use[3])
        nn5_size.append(sizes_to_use[4])

        # Unpack distances
        nn1_dist.append(dists_to_use[0])
        nn2_dist.append(dists_to_use[1])
        nn3_dist.append(dists_to_use[2])
        nn4_dist.append(dists_to_use[3])
        nn5_dist.append(dists_to_use[4])

    # Assign everything back to the dataframe
    gdf["nn1_garden_size"] = nn1_size
    gdf["nn2_garden_size"] = nn2_size
    gdf["nn3_garden_size"] = nn3_size
    gdf["nn4_garden_size"] = nn4_size
    gdf["nn5_garden_size"] = nn5_size

    gdf["nn1_distance_m"] = nn1_dist
    gdf["nn2_distance_m"] = nn2_dist
    gdf["nn3_distance_m"] = nn3_dist
    gdf["nn4_distance_m"] = nn4_dist
    gdf["nn5_distance_m"] = nn5_dist

    if "coord_round" in gdf.columns:
        gdf = gdf.drop(columns=["coord_round"])

    return gdf


def _get_gdf_number_uprns_within_radius(gdf, radius_m=100):
    """
    Find the number of UPRNs within a given radius of UPRN point coordinates.
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing UPRN point coordinates. Must contain a 'UPRN' column
        radium_m (float): distance (m) to buffer around each point (default 100m)
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


def _calculate_gdf_plot_ratio_proxy(buildings_gdf, buffer_radius=100):
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


def _get_int_count_vertices(geom):
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


def _compute_voronoi_area(gdf, grid_squares, boundary):
    """
    Calculates the area within voronoi polygons formed for each UPRN coordinate, with barrier features removed
    Args:
        gdf (gpd.GeoDataFrame): UPRN point coordinates
        grid_squares (str): grid squares to get the barrier features for
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


def engineer_gdf_features(local_authorities):
    """
    Engineer set of features for model training at the UPRN level
    Args:
        local_authorities (str | list[str]): Local Authority or Authorities to engineer features for

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
    uprn_counts = uprns_df.groupby("ID").size().reset_index(name="n_uprns_in_building")
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
    gdf_with_features = _get_gdf_5nn_spatial_features(
        gdf=gdf_with_features, unique_id_col="UPRN"
    )

    uprn_sums = _get_gdf_number_uprns_within_radius(gdf=gdf_with_features)
    gdf_with_features = gdf_with_features.merge(uprn_sums, on="UPRN", how="left")

    return gdf_with_features
