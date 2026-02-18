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
import momepy as mm

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
).to_crs(
    epsg="27700"
)  # Convert to British National Grid

# %%
tech_gdf["tech"].unique()

# %%
TECHS = [
    "Individual solution",
    "Networked GSHP",
    "Communal solutions",
    "District heat network",
    "Individual solution or Networked GSHP",
    "Individual solution or District heat network",
]

COLOURS = {
    "Individual solution": "#18A48C",
    "Networked GSHP": "#0000FF",
    "Communal solutions": "#FF6E47",
    "District heat network": "#EA2541",
    # "Individual solution or Networked GSHP": "grey",
    # "Individual solution or District heat network": "gray",
}

# %%
# Replace tech type with district heat network if in heat network zone
tech_gdf["tech"] = tech_gdf.apply(
    lambda x: ("District heat network" if x["in_hn_zone"] else x["tech"]),
    axis=1,
)

tech_gdf["color"] = tech_gdf.apply(
    lambda x: "#EA2541" if x["in_hn_zone"] else x["color"], axis=1
)

# %%


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

    Returns:
        folium.Map: interactive map with dissolved polygons and buildings plotted
        gpd.GeoDataFrame: dissolved geodataframe with tech types and geometries
    """
    # Dissolve by tech type
    dissolved_gdf = voronoi_gdf.dissolve(by="tech").reset_index()
    dissolved_gdf["colour"] = dissolved_gdf["tech"].map(colours)

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    dissolved_4326_gdf = dissolved_gdf.to_crs(epsg=4326)
    buildings_gdf = buildings_gdf.to_crs(epsg=4326)
    # dissolved_4326_gdf['geometry'] = dissolved_4326_gdf['geometry'].simplify(tolerance=0.000005, preserve_topology=True)

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
                "color": "black",
                "weight": 0.5,
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
            style_function=lambda x: {
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0,
            },
        )
        geo_j.add_to(m)

    return m, dissolved_4326_gdf


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
# New code - still Gemini generated. Needs review and lots of tweaks still
from shapely.geometry import MultiPoint, box, Polygon, MultiPolygon
from shapely.ops import voronoi_diagram


def extend_edges(gdf, segment_distance=1.0, max_edge_dist=20.0):
    """
    High-performance version using vectorized overlay for clipping.
    """
    gdf = gdf.copy()
    uid_col = "_internal_unique_id"
    gdf[uid_col] = np.arange(len(gdf))

    all_points = []
    all_ids = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        src_id = row[uid_col]
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        for poly in polys:
            if not isinstance(poly, Polygon):
                continue

            # Densify all rings (exterior + all interiors)
            rings = [poly.exterior] + list(poly.interiors)
            for ring in rings:
                num_pts = int(np.ceil(ring.length / segment_distance))
                # Use list comprehension for faster point generation
                pts = [ring.interpolate(i * segment_distance) for i in range(num_pts)]
                all_points.extend(pts)
                all_ids.extend([src_id] * len(pts))

    points_gdf = gpd.GeoDataFrame({uid_col: all_ids}, geometry=all_points, crs=gdf.crs)

    coords = MultiPoint(points_gdf.geometry.tolist())
    # Envelope is slightly larger than the 20m buffer limit
    vor_collection = voronoi_diagram(
        coords, envelope=box(*gdf.total_bounds).buffer(max_edge_dist + 5)
    )
    vor_gdf = gpd.GeoDataFrame(geometry=list(vor_collection.geoms), crs=gdf.crs)

    # We join points to Voronoi cells and dissolve to get one polygon per building ID
    joined = gpd.sjoin(vor_gdf, points_gdf, how="inner", predicate="contains")
    final_voronoi = joined.dissolve(by=uid_col).reset_index()

    # Create the 20m constraint buffers for all buildings at once
    constraint_buffers = gdf[[uid_col, "geometry"]].copy()
    constraint_buffers["geometry"] = constraint_buffers.geometry.buffer(max_edge_dist)

    # Use overlay 'intersection' to clip cells by their respective buffers
    # We filter the result to ensure building A's Voronoi is only clipped by building A's buffer
    result_gdf = gpd.overlay(final_voronoi, constraint_buffers, how="intersection")
    result_gdf = result_gdf[result_gdf[f"{uid_col}_1"] == result_gdf[f"{uid_col}_2"]]

    # Cleanup and merge original attributes
    result_gdf = result_gdf.rename(columns={f"{uid_col}_1": uid_col})
    result_gdf = result_gdf.drop(columns=[f"{uid_col}_2"])

    final_result = result_gdf.merge(gdf.drop(columns="geometry"), on=uid_col)
    return final_result.drop(columns=[uid_col])


# %%
# Old code
# def extend_edges(gdf, segment_distance=1.0, buffer_factor=2.0):
#     """
#     Extends the edges of polygons to create a Voronoi diagram filling the surrounding space.
#     Rewritten logic by Gemini based on fieldmaps/edge-extender.

#     Args:
#         gdf (gpd.GeoDataFrame): Input GeoDataFrame containing polygons.
#         segment_distance (float): Distance (in CRS units) to interval points along edges.
#         buffer_factor (float): Factor to expand the bounding box for the Voronoi extent.

#     Returns:
#         gpd.GeoDataFrame: A GeoDataFrame of the extended Voronoi regions.
#     """

#     # Work on a copy to avoid modifying the original
#     gdf = gdf.copy()

#     # 1. Create a guaranteed unique internal ID for tracking
#     # We use a temporary string column name that is unlikely to conflict with user data
#     uid_col = "_internal_unique_id"
#     gdf[uid_col] = np.arange(len(gdf))

#     print("1. Preparing boundaries...")
#     # Calculate the union of all polygons to find the outer boundary
#     dissolved = gdf.unary_union

#     # Extract the boundary of the union (lines touching the empty space)
#     if hasattr(dissolved, "boundary"):
#         overall_boundary = dissolved.boundary
#     else:
#         raise ValueError("Could not compute boundary of input polygons.")

#     # Create a GDF for the boundary lines
#     boundary_gdf = gpd.GeoDataFrame(geometry=[overall_boundary], crs=gdf.crs)
#     boundary_gdf = boundary_gdf.explode(index_parts=False)

#     # Intersect boundary with original polygons to attribute the edges
#     # We keep only the segments that overlap with our original polygons
#     # This attaches our 'uid_col' to the edge segments
#     attributed_edges = gpd.overlay(
#         boundary_gdf,
#         gdf[[uid_col, "geometry"]],
#         how="intersection",
#         keep_geom_type=True,
#     )

#     # Filter for linear geometries only
#     attributed_edges = attributed_edges[
#         attributed_edges.geometry.type.isin(["LineString", "MultiLineString"])
#     ]

#     print("2. Generating points from edges...")
#     all_points = []
#     all_ids = []

#     # Iterate through the edges to densify them into points
#     for _, row in attributed_edges.iterrows():
#         geom = row.geometry
#         src_id = row[uid_col]

#         if geom.is_empty:
#             continue

#         # Standardize to list of lines
#         lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]

#         for line in lines:
#             coords = list(line.coords)

#             # Interpolate points for long segments
#             length = line.length
#             if length > segment_distance:
#                 num_segments = int(np.ceil(length / segment_distance))
#                 # Generate intermediate points (skipping 0 to avoid duplication with vertices)
#                 for i in range(1, num_segments):
#                     pt = line.interpolate(i * segment_distance)
#                     all_points.append(pt)
#                     all_ids.append(src_id)

#             # Add original vertices to preserve corners
#             for coord in coords:
#                 all_points.append(Point(coord))
#                 all_ids.append(src_id)

#     # Create GDF of points linked to the internal ID
#     points_gdf = gpd.GeoDataFrame({uid_col: all_ids}, geometry=all_points, crs=gdf.crs)

#     print(f"   Generated {len(points_gdf)} points.")

#     print("3. Computing Voronoi diagram...")
#     # Define bounding box for Voronoi extent
#     minx, miny, maxx, maxy = gdf.total_bounds
#     width = maxx - minx
#     height = maxy - miny
#     envelope = box(
#         minx - width * buffer_factor,
#         miny - height * buffer_factor,
#         maxx + width * buffer_factor,
#         maxy + height * buffer_factor,
#     )

#     # Generate Voronoi regions
#     if len(points_gdf) == 0:
#         raise ValueError(
#             "No boundary points generated. Are polygons touching/overlapping perfectly?"
#         )

#     multi_point = MultiPoint(points_gdf.geometry.tolist())
#     voronoi_geoms = voronoi_diagram(multi_point, envelope=envelope)

#     # Convert GeometryCollection to list of Polygons
#     voronoi_polys = list(voronoi_geoms.geoms)
#     voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_polys, crs=gdf.crs)

#     print("4. Linking Voronoi regions to original polygons...")
#     # Spatial join to link Voronoi polygons back to the generating points (and thus the ID)
#     joined = gpd.sjoin(voronoi_gdf, points_gdf, how="inner", predicate="contains")

#     print("5. Dissolving regions...")
#     # Dissolve Voronoi cells by the internal ID
#     final_voronoi = joined.dissolve(by=uid_col, as_index=False)

#     # Merge original attributes back using the internal ID
#     # We drop the geometry from the original merge to keep the new Voronoi geometry
#     original_attrs = gdf.drop(columns="geometry")
#     result = final_voronoi.merge(original_attrs, on=uid_col)

#     # Clean up temporary column
#     result = result.drop(columns=[uid_col])

#     return result


# # Example Usage:
# # polygons = gpd.read_file("my_polygons.shp")
# # # Ensure you are in a projected CRS (meters) for segment_distance to make sense
# # polygons = polygons.to_crs(epsg=3857)
# # voronoi = extend_edges(polygons, segment_distance=50)
# # voronoi.to_file("voronoi_output.gpkg")

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

# %% [markdown]
# ## Add boundaries to polygons

# %%
# Linestrings
roads_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="road", grid_squares="SX"
)
railways_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="railway_track", grid_squares="SX"
)
tidal_boundary_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="tidal_boundary", grid_squares="SX"
)

# Polygons
forest_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="woodland", grid_squares="SX"
)
greenspace_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/geodata/v202510_OSOpenMapGreenspace_geometries_selected/SX/SX_GreenspaceSite.shp"
)
surface_water_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/geodata/v202510_OSOpenMapLocal_geometries_selected/SX/SX_SurfaceWater_Area.shp"
)
tidal_water_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="tidal_water", grid_squares="SX"
)

# %%
line_overlays = [roads_gdf, railways_gdf]
polygon_overlays = [forest_gdf, greenspace_gdf, tidal_water_gdf, surface_water_gdf]

line_overlay_gdf = pd.concat([gdf[["geometry"]] for gdf in line_overlays])
line_overlay_gdf["geometry"] = line_overlay_gdf.geometry.buffer(1.75)
polygon_overlay_gdf = pd.concat([gdf[["geometry"]] for gdf in polygon_overlays])

tessellation_gdf = (
    edge_extend_gdf.overlay(polygon_overlay_gdf, how="difference")
    .overlay(line_overlay_gdf, how="difference")
    .drop(columns="index_right")
    .explode()
)
tessellation_gdf = (
    tessellation_gdf.sjoin(tech_gdf, how="inner", predicate="intersects")
    .drop(columns=["index_right", "tech_right"])
    .rename(columns={"tech_left": "tech"})
)

# %%
tessellation_gdf.plot()

# %%
map, dissolved_gdf = dissolve_techs_and_plot_folium(
    voronoi_gdf=tessellation_gdf,
    buildings_gdf=tech_gdf,
    boundary_gdf=plymouth_boundaries,
)
map

# %%
map.save("20250205_plymouth_folium_map.html")

# %%


# %%
dissolved_gdf.to_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons.geojson",
    driver="GeoJSON",
)

# %%


# %%
# # To open this geojson s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons.geojson

# saved_gdf = gpd.read_file(
#     "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons.geojson"
# )


# %%
