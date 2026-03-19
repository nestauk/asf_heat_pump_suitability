"""
Functions to generate features for random forest binary classifier model which classifies buildings into blocks of flats
or not. Features are generated per building from building footprint and UPRN geodata.
"""

import numpy as np
import polars as pl
import geopandas as gpd
import pandas as pd
import shapely

from asf_heat_pump_suitability.pipeline.transform import building_footprints


def generate_df_features(
    buildings_gdf: gpd.GeoDataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    id_col: str,
    boundaries_gdf: gpd.GeoDataFrame,
) -> pl.DataFrame:
    """
    Generate all features required to train block of flats random forest binary classifier.

    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprints of buildings containing at least one domestic property.
        uprns_gdf (gpd.GeoDataFrame): all UPRNs (both domestic and non-domestic) located within the footprints in `buildings_gdf` with corresponding geospatial points and `property_type_flat` boolean column.
        id_col (str): name of building ID column

    Return:
        pl.DataFrame: all features for each building ID (per id_col)
    """
    print("Generating features required for block of flats classifier...")
    # Join UPRNs to the building footprints they intersect with and retain building geometry
    buildings_w_uprns_gdf = buildings_gdf.sjoin(
        uprns_gdf, how="inner", predicate="intersects"
    ).dropna(subset=id_col)

    # Join buildings to the UPRNs they contain and retain UPRN geometry
    uprns_w_buildings_gdf = uprns_gdf.sjoin(
        buildings_gdf, how="inner", predicate="intersects"
    ).dropna(subset="UPRN")

    features_dfs = [
        # Generate features per building footprint
        _generate_df_building_features(buildings_w_uprns_gdf, id_col),
        _generate_df_stacked_uprn_features(uprns_w_buildings_gdf, id_col),
        _generate_df_concave_hull_features(uprns_w_buildings_gdf, id_col),
        _generate_df_convex_hull_features(uprns_w_buildings_gdf, id_col),
        _generate_df_footprint_geometry_features(buildings_gdf, id_col),
        _generate_df_uprn_perimeter_distance_features(buildings_w_uprns_gdf, id_col),
        _generate_df_building_sections_features(
            uprns_gdf=uprns_gdf,
            buildings_gdf=buildings_gdf,
            boundaries_gdf=boundaries_gdf,
            id_col=id_col,
        ),
    ]

    features_dfs = pl.align_frames(*features_dfs, on=id_col, how="left")

    return pl.concat(features_dfs, how="align").with_columns(
        (pl.col("concave_hull_area_m2") / pl.col("building_area_m2")).alias(
            "hull_to_building_area_ratio"
        ),
        (pl.col("convex_hull_area_m2") / pl.col("building_area_m2")).alias(
            "convex_hull_to_building_area_ratio"
        ),
        pl.when(pl.col("convex_hull_area_m2") == 0)
        .then(-1)
        .otherwise(pl.col("concave_hull_area_m2") / pl.col("convex_hull_area_m2"))
        .alias("concave_to_convex_hull_ratio"),
    )


def _generate_df_building_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """
    Generate select features from building footprints with UPRNs joined to them:
    - n_UPRNs (per building)
    - n_flats (per building)
    - building_area_m2
    - building_perimeter_m
    - proportion_flats (proportion of UPRNs which are flats)
    - UPRNs_per_building_m2 (UPRN density per m2 of building footprint)

    Args:
        gdf (gpd.GeoDataFrame): building footprints with UPRNs joined to them. UPRNs must have `property_type_flat` boolean column.
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint (per id_col)
    """
    gdf["building_area_m2"] = gdf.area
    gdf["building_perimeter_m"] = gdf.length

    df = pl.from_pandas(gdf.drop(columns=["geometry"]))

    # Aggregate data per building
    agg_building_df = (
        df.group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
            pl.col("building_area_m2").first().name.keep(),
            pl.col("building_perimeter_m").first().name.keep(),
        )
        .with_columns(
            (pl.col("n_flats") / pl.col("n_UPRNs")).alias("proportion_flats"),
            (pl.col("n_UPRNs") / pl.col("building_area_m2")).alias(
                "UPRNs_per_building_m2"
            ),
        )
    )

    return agg_building_df


def _generate_df_stacked_uprn_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate select features from UPRNs with building footprints joined to them:
    - avg_n_stacked_uprns (the average number of UPRNs sharing the same coordinates per building)
    - std_n_stacked_uprns (the standard deviation of the number of UPRNs sharing the same coordinates per building)
    - max_n_stacked_uprns (the maximum number of UPRNs sharing the same coordinates per building)

    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint (per id_col)
    """
    # Get count of UPRNs at each X and Y coordinates to get the count of UPRNs which share an exact location
    df = pl.from_pandas(gdf.drop(columns="geometry"))
    df = df.with_columns(
        # Count of stacked UPRNs per coordinate
        n_stacked_uprns=pl.col("UPRN")
        .count()
        .over(["X_COORDINATE", "Y_COORDINATE"])
    )

    # Group by building and get the average, STD, and max of UPRNs sharing the same coordinates
    df = df.group_by(id_col).agg(
        pl.col("n_stacked_uprns").mean().alias("avg_n_stacked_uprns"),
        pl.col("n_stacked_uprns").std().alias("std_n_stacked_uprns"),
        pl.col("n_stacked_uprns").max().alias("max_n_stacked_uprns"),
    )

    return df


def _generate_df_concave_hull_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate select features from UPRNs with building footprints joined to them:
    - concave_hull_area_m2 (the area (m2) of the concave hull of the point geometries of all UPRNs per building)
    - uprns_per_hull_area_m2 (the number of UPRNs per concave hull area per building)
    - flats_per_hull_area_m2 (the number of flats per concave hull area per building)

    The concave hull is a representation of the spread of UPRNs within the building footprint.

    Note: UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2).
    In these cases, the uprns_ or flats_per_hull_area_m2 is changed to -1.

    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint (per id_col)
    """
    # Create concave hull feature to represent spatial distribution of UPRNs in each building
    hull_gdf = gdf.dissolve(id_col).concave_hull().reset_index()
    hull_gdf = hull_gdf.rename(columns={0: "geometry"}).set_geometry("geometry")
    hull_gdf["concave_hull_area_m2"] = hull_gdf.area

    # Aggregate data per building
    agg_building_df = (
        pl.from_pandas(gdf.drop(columns=["geometry"]))
        .group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
        )
        .join(
            # Join building features with concave hull feature
            pl.from_pandas(hull_gdf.drop(columns="geometry")),
            how="left",
            on=id_col,
        )
        .with_columns(
            # Calculate additional features from the concave hull area
            (pl.col("n_UPRNs") / pl.col("concave_hull_area_m2")).alias(
                "uprns_per_hull_area_m2"
            ),
            (pl.col("n_flats") / pl.col("concave_hull_area_m2")).alias(
                "flats_per_hull_area_m2"
            ),
        )
        .with_columns(
            # UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2)
            # We change this to -1 for the model
            pl.when(pl.col("uprns_per_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("uprns_per_hull_area_m2"))
            .alias("uprns_per_hull_area_m2"),
            pl.when(pl.col("flats_per_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("flats_per_hull_area_m2"))
            .alias("flats_per_hull_area_m2"),
        )
    )

    keep_cols = [
        id_col,
        "concave_hull_area_m2",
        "uprns_per_hull_area_m2",
        "flats_per_hull_area_m2",
    ]

    return agg_building_df.select(keep_cols)


def _generate_df_convex_hull_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate convex hull features from UPRNs with building footprints joined to them:
    - convex_hull_area_m2 (the area (m2) of the convex hull of the point geometries of all UPRNs per building)
    - uprns_per_convex_hull_area_m2 (the number of UPRNs per convex hull area per building)
    - flats_per_convex_hull_area_m2 (the number of flats per convex hull area per building)

    The convex hull is the smallest convex polygon containing all UPRNs in the building. Combined
    with the concave hull, the ratio of the two provides information on the shape regularity of the
    UPRN distribution (a ratio close to 1 means UPRNs are spread in a convex pattern).

    Note: UPRNs or flats per hull area can be infinite if all UPRNs share collinear or identical
    coordinates (i.e. area = 0m2). In these cases the value is set to -1.

    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint (per id_col)
    """
    dissolved = gdf.dissolve(id_col)
    dissolved["convex_hull_area_m2"] = dissolved.geometry.convex_hull.area
    hull_df = pl.from_pandas(dissolved[["convex_hull_area_m2"]].reset_index())

    agg_building_df = (
        pl.from_pandas(gdf.drop(columns=["geometry"]))
        .group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
        )
        .join(hull_df, how="left", on=id_col)
        .with_columns(
            (pl.col("n_UPRNs") / pl.col("convex_hull_area_m2")).alias(
                "uprns_per_convex_hull_area_m2"
            ),
            (pl.col("n_flats") / pl.col("convex_hull_area_m2")).alias(
                "flats_per_convex_hull_area_m2"
            ),
        )
        .with_columns(
            pl.when(pl.col("uprns_per_convex_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("uprns_per_convex_hull_area_m2"))
            .alias("uprns_per_convex_hull_area_m2"),
            pl.when(pl.col("flats_per_convex_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("flats_per_convex_hull_area_m2"))
            .alias("flats_per_convex_hull_area_m2"),
        )
    )

    return agg_building_df.select(
        [
            id_col,
            "convex_hull_area_m2",
            "uprns_per_convex_hull_area_m2",
            "flats_per_convex_hull_area_m2",
        ]
    )


def _generate_df_footprint_geometry_features(
    buildings_gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate geometric descriptor features directly from building footprint polygons:
    - n_building_vertices (total number of exterior ring vertices across all polygon parts)
    - footprint_edge_ratio (ratio of longest to shortest exterior edge; None for degenerate geometries)

    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprints, one row per building
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint (per id_col)
    """

    def _polygon_edges(geom) -> np.ndarray:
        """Return edge lengths for all exterior rings in a Polygon or MultiPolygon."""
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        edges = []
        for part in parts:
            coords = np.array(part.exterior.coords)
            edges.append(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        return np.concatenate(edges)

    n_vertices = []
    edge_ratios = []

    for geom in buildings_gdf.geometry:
        if geom.geom_type == "Polygon":
            n_verts = len(geom.exterior.coords) - 1
        elif geom.geom_type == "MultiPolygon":
            n_verts = sum(len(g.exterior.coords) - 1 for g in geom.geoms)
        else:
            n_verts = 0

        n_vertices.append(n_verts)

        if geom.geom_type in ("Polygon", "MultiPolygon"):
            edge_lengths = _polygon_edges(geom)
            min_edge = edge_lengths.min()
            edge_ratios.append(edge_lengths.max() / min_edge if min_edge > 0 else None)
        else:
            edge_ratios.append(None)

    return pl.DataFrame(
        {
            id_col: buildings_gdf[id_col].tolist(),
            "n_building_vertices": n_vertices,
            "footprint_edge_ratio": edge_ratios,
        }
    )


def _generate_df_uprn_perimeter_distance_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Compute the distance from each UPRN to its building's exterior perimeter, then aggregate
    per building:
    - avg_uprn_perimeter_dist_m
    - std_uprn_perimeter_dist_m
    - min_uprn_perimeter_dist_m
    - max_uprn_perimeter_dist_m

    UPRNs near the perimeter (low distance) suggest edge-of-building units; UPRNs far from the
    perimeter suggest interior units — a pattern useful for distinguishing block of flats from
    other building types.

    Args:
        gdf (gpd.GeoDataFrame): building footprints with UPRNs joined to them (building geometry
            is the active geometry; UPRN coordinates come from X_COORDINATE / Y_COORDINATE columns)
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: distance summary features per building footprint (per id_col)
    """
    uprn_points = shapely.points(gdf["X_COORDINATE"].values, gdf["Y_COORDINATE"].values)
    # shapely.boundary() returns the exterior ring for Polygon or MultiLineString for MultiPolygon
    building_boundaries = shapely.boundary(gdf.geometry.values)
    dists = shapely.distance(uprn_points, building_boundaries)

    return (
        pl.DataFrame({id_col: gdf[id_col].tolist(), "uprn_perimeter_dist_m": dists})
        .group_by(id_col)
        .agg(
            pl.col("uprn_perimeter_dist_m").mean().alias("avg_uprn_perimeter_dist_m"),
            pl.col("uprn_perimeter_dist_m").std().alias("std_uprn_perimeter_dist_m"),
            pl.col("uprn_perimeter_dist_m").min().alias("min_uprn_perimeter_dist_m"),
            pl.col("uprn_perimeter_dist_m").max().alias("max_uprn_perimeter_dist_m"),
        )
    )


def _generate_df_building_sections_features(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    id_col: str,
) -> pl.DataFrame:
    """ """
    uprns_gdf = (
        uprns_gdf.sjoin(
            boundaries_gdf[["LAD23NM", "geometry"]], how="inner", predicate="intersects"
        )
        .drop_duplicates(subset="UPRN")
        .drop(columns="index_right")
    )

    gdfs = []
    for local_authority in uprns_gdf["LAD23NM"].unique():
        _uprns_gdf = uprns_gdf[uprns_gdf["LAD23NM"] == local_authority]
        boundary = boundaries_gdf[boundaries_gdf["LAD23NM"] == local_authority][
            "geometry"
        ].values[0]
        _building_units_gdf = building_footprints.generate_gdf_building_sections(
            uprns_gdf=_uprns_gdf, buildings_gdf=buildings_gdf, boundary=boundary
        )
        gdfs.append(_building_units_gdf)

    building_units_gdf = pd.concat(gdfs)
    building_units_gdf["building_unit_area_m2"] = building_units_gdf.area
    building_units_gdf["building_unit_perimeter_m2"] = building_units_gdf.length

    return (
        pl.from_pandas(building_units_gdf.drop(columns="geometry"))
        .group_by(id_col)
        .agg(
            pl.col("n_UPRNs").count().alias("n_building_units"),
            pl.col("building_unit_area_m2").mean().alias("avg_building_unit_area_m2"),
            pl.col("building_unit_perimeter_m2")
            .mean()
            .alias("avg_building_unit_perimeter_m2"),
            pl.col("n_UPRNs").mean().alias("avg_n_uprns_per_building_unit"),
        )
    )
