# %% [markdown]
# ## First pass for merging polygons
#
# This notebook explores the use of Voronoi diagrams to create contiguous polygons to fill the space within a given boundary (Plymouth). The starting point for the merged polygons are the building footprint polygons after they have been labelled with a suitable tech type. Neighbouring buildings which have the same technology should end up in the same merged polygon.

# %%
import polars as pl
import pandas as pd
import numpy as np

import geopandas as gpd
import shapely
from shapely.geometry import MultiPoint, LineString, Point, box
from shapely.ops import voronoi_diagram, unary_union

import folium

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from asf_heat_pump_suitability.getters import load_tree_input, load_boundaries

# %%
# Load Plymouth LA boundary
plymouth_boundaries = load_boundaries.load_gdf_local_authority_boundaries("plymouth")
plymouth_boundary = plymouth_boundaries.geometry.values[0]

# Load tech type per building
tech_gdf = (
    gpd.read_parquet(
        "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.parquet"
    )
    .rename(columns={"1st_most_suitable_solution": "tech"})
    .dropna(subset="geometry")
    .drop_duplicates(subset="geometry")
)

# %%
TECHS = [
    "Communal solutions",
    "District heat network",
    "Individual solution",
    "Networked GSHP",
]

COLOURS = {
    "Individual solution": "orange",
    "Networked GSHP": "green",
    "Communal solutions": "hotpink",
    "District heat network": "blue",
}

# %% [markdown]
# ## Plotting functions


# %%
def plot_tech_separate_subplots(
    gdf: gpd.GeoDataFrame, colours: dict = COLOURS, techs: list[str] = TECHS
) -> None:
    """
    Plot polygons separately for each tech type across different subplots.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting
        techs (list[str]): list of tech types

    Returns None
    """
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    for ax, tech in zip(axs.ravel(), techs):
        gdf[gdf["tech"] == tech].plot(color=colours[tech], ax=ax)
        ax.set_title(tech)

    fig.suptitle("Spatial distribution of tech types in Plymouth Local Authority")
    plt.tight_layout()


def dissolve_techs_and_plot_static(gdf: gpd.GeoDataFrame, colours: dict = COLOURS):
    """
    Plot polygons together for each tech type in a static chart.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns None
    """
    dissolved_gdf = gdf.dissolve(by="tech").reset_index()
    dissolved_gdf["colour"] = dissolved_gdf["tech"].map(colours)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    dissolved_gdf.plot(
        color=dissolved_gdf["colour"],
        ax=ax,
        legend_kwds={"labels": dissolved_gdf["tech"]},
        legend=True,
    )

    patches = [
        mpatches.Patch(facecolor=colour, edgecolor=colour, label=tech)
        for tech, colour in colours.items()
    ]
    fig.suptitle("Spatial distribution of tech types in Plymouth Local Authority")
    plt.legend(handles=patches)
    plt.show()


def dissolve_techs_and_plot_folium(
    voronoi_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    colours: dict = COLOURS,
):
    """
    Plot the resulting polygons together for each tech type in Folium.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns Folium map
    """
    # Dissolve by tech type
    dissolved_gdf = voronoi_gdf.dissolve(by="tech").reset_index()
    dissolved_gdf["colour"] = dissolved_gdf["tech"].map(colours)

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    dissolved_4326_gdf = dissolved_gdf.to_crs(epsg=4326)
    buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    # Get centre of boundary to centre map
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)["geometry"].values[0]
    centre_map = shapely.get_coordinates(boundary_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    # Plot voronoi polygons
    for _, r in dissolved_4326_gdf.iterrows():
        colour = colours[r["tech"]]
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour=colour: {
                "fillColor": colour,
                "fillOpacity": 0.66,
            },
        )
        folium.Popup(r["tech"]).add_to(geo_j)
        geo_j.add_to(m)

    # Plot buildings
    for _, r in buildings_gdf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {"color": "black", "weight": 1, "fillOpacity": 0},
        )
        geo_j.add_to(m)

    return m


# %% [markdown]
# ## Test voronoi polygons on building vertices

# %%
# Get Voronoi polygons of building vertices
voronoi_polygons_gdf = gpd.GeoDataFrame(
    {"geometry": tech_gdf.geometry.voronoi_polygons(extend_to=plymouth_boundary)},
    geometry="geometry",
    crs="27700",
)

# Join them back to the vertices they cover
voronoi_polygons_gdf = voronoi_polygons_gdf.sjoin(
    tech_gdf, how="left", predicate="intersects"
).clip(plymouth_boundary)

# %%
plot_tech_separate_subplots(voronoi_polygons_gdf)

# %%
dissolve_techs_and_plot_static(voronoi_polygons_gdf)

# %%
dissolve_techs_and_plot_folium(
    voronoi_gdf=voronoi_polygons_gdf,
    buildings_gdf=tech_gdf,
    boundary_gdf=plymouth_boundaries,
)

# %% [markdown]
# ## Densify edges and then voronoi


# %%
def extend_edges(gdf, segment_distance=1.0, buffer_factor=2.0):
    """
    Extends the edges of polygons to create a Voronoi diagram filling the surrounding space.
    Rewritten logic by Gemini based on fieldmaps/edge-extender.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing polygons.
        segment_distance (float): Distance (in CRS units) to interval points along edges.
        buffer_factor (float): Factor to expand the bounding box for the Voronoi extent.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of the extended Voronoi regions.
    """

    # Work on a copy to avoid modifying the original
    gdf = gdf.copy()

    # 1. Create a guaranteed unique internal ID for tracking
    # We use a temporary string column name that is unlikely to conflict with user data
    uid_col = "_internal_unique_id"
    gdf[uid_col] = np.arange(len(gdf))

    print("1. Preparing boundaries...")
    # Calculate the union of all polygons to find the outer boundary
    dissolved = gdf.unary_union

    # Extract the boundary of the union (lines touching the empty space)
    if hasattr(dissolved, "boundary"):
        overall_boundary = dissolved.boundary
    else:
        raise ValueError("Could not compute boundary of input polygons.")

    # Create a GDF for the boundary lines
    boundary_gdf = gpd.GeoDataFrame(geometry=[overall_boundary], crs=gdf.crs)
    boundary_gdf = boundary_gdf.explode(index_parts=False)

    # Intersect boundary with original polygons to attribute the edges
    # We keep only the segments that overlap with our original polygons
    # This attaches our 'uid_col' to the edge segments
    attributed_edges = gpd.overlay(
        boundary_gdf,
        gdf[[uid_col, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    # Filter for linear geometries only
    attributed_edges = attributed_edges[
        attributed_edges.geometry.type.isin(["LineString", "MultiLineString"])
    ]

    print("2. Generating points from edges...")
    all_points = []
    all_ids = []

    # Iterate through the edges to densify them into points
    for _, row in attributed_edges.iterrows():
        geom = row.geometry
        src_id = row[uid_col]

        if geom.is_empty:
            continue

        # Standardize to list of lines
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:
            coords = list(line.coords)

            # Interpolate points for long segments
            length = line.length
            if length > segment_distance:
                num_segments = int(np.ceil(length / segment_distance))
                # Generate intermediate points (skipping 0 to avoid duplication with vertices)
                for i in range(1, num_segments):
                    pt = line.interpolate(i * segment_distance)
                    all_points.append(pt)
                    all_ids.append(src_id)

            # Add original vertices to preserve corners
            for coord in coords:
                all_points.append(Point(coord))
                all_ids.append(src_id)

    # Create GDF of points linked to the internal ID
    points_gdf = gpd.GeoDataFrame({uid_col: all_ids}, geometry=all_points, crs=gdf.crs)

    print(f"   Generated {len(points_gdf)} points.")

    print("3. Computing Voronoi diagram...")
    # Define bounding box for Voronoi extent
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    envelope = box(
        minx - width * buffer_factor,
        miny - height * buffer_factor,
        maxx + width * buffer_factor,
        maxy + height * buffer_factor,
    )

    # Generate Voronoi regions
    if len(points_gdf) == 0:
        raise ValueError(
            "No boundary points generated. Are polygons touching/overlapping perfectly?"
        )

    multi_point = MultiPoint(points_gdf.geometry.tolist())
    voronoi_geoms = voronoi_diagram(multi_point, envelope=envelope)

    # Convert GeometryCollection to list of Polygons
    voronoi_polys = list(voronoi_geoms.geoms)
    voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_polys, crs=gdf.crs)

    print("4. Linking Voronoi regions to original polygons...")
    # Spatial join to link Voronoi polygons back to the generating points (and thus the ID)
    joined = gpd.sjoin(voronoi_gdf, points_gdf, how="inner", predicate="contains")

    print("5. Dissolving regions...")
    # Dissolve Voronoi cells by the internal ID
    final_voronoi = joined.dissolve(by=uid_col, as_index=False)

    # Merge original attributes back using the internal ID
    # We drop the geometry from the original merge to keep the new Voronoi geometry
    original_attrs = gdf.drop(columns="geometry")
    result = final_voronoi.merge(original_attrs, on=uid_col)

    # Clean up temporary column
    result = result.drop(columns=[uid_col])

    return result


# Example Usage:
# polygons = gpd.read_file("my_polygons.shp")
# # Ensure you are in a projected CRS (meters) for segment_distance to make sense
# polygons = polygons.to_crs(epsg=3857)
# voronoi = extend_edges(polygons, segment_distance=50)
# voronoi.to_file("voronoi_output.gpkg")

# %%
edge_extend_gdf = extend_edges(tech_gdf).clip(plymouth_boundary)

# %%
plot_tech_separate_subplots(edge_extend_gdf)

# %%
dissolve_techs_and_plot_static(edge_extend_gdf)

# %%
dissolve_techs_and_plot_folium(
    voronoi_gdf=edge_extend_gdf,
    buildings_gdf=tech_gdf,
    boundary_gdf=plymouth_boundaries,
)
