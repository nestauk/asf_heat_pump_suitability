# %%
import polars as pl
import geopandas as gpd

# %%
cluster_geometries = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons_with_clusterID.geojson"
)

# %%
cluster_geometries["code"].unique()

# %%
cluster_contextual_info = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth/plymouth_cluster_contextual_features.geojson"
)

# %%
cluster_contextual_info = cluster_contextual_info.sort_values("cluster_id")

# %%
cluster_contextual_info.columns

# %%
null_cols = [col for col in cluster_contextual_info.columns if "null" in col.lower()]

# %%
cluster_contextual_info["nulls"] = cluster_contextual_info[null_cols].sum(axis=1)

# %%
clusters_to_remove = cluster_contextual_info[cluster_contextual_info["nulls"] > 0][
    "cluster_id"
].unique()

# %%
cluster_geometries = cluster_geometries[
    ~cluster_geometries["cluster_id"].isin(clusters_to_remove)
]
cluster_contextual_info = cluster_contextual_info[
    ~cluster_contextual_info["cluster_id"].isin(clusters_to_remove)
]

# %%
cluster_geometries.crs

# %%
cluster_contextual_info.crs

# %%
cluster_contextual_info.drop(columns=null_cols, inplace=True)

# %%
cluster_geometries.to_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons_with_clusterID.geojson",
    driver="GeoJSON",
)

# %%
cluster_contextual_info.to_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth/plymouth_cluster_contextual_features.geojson",
    driver="GeoJSON",
)

# %%
import geopandas as gpd
import shapely

import folium
import momepy as mm

import matplotlib.pyplot as plt

# %%

boundary_4326 = cluster_geometries["geometry"].values[0]
centre_map = shapely.get_coordinates(boundary_4326.centroid)

colours = {
    "Individual solution": "#18A48C",
    "Networked Heat Pump": "#0000FF",
    "Communal solutions": "#FF6E47",
    "District heat network": "#EA2541",
}

# Create map
m = folium.Map(
    location=[centre_map[0][1], centre_map[0][0]],
    zoom_start=15,
    tiles="esri_worldimagery",
)

# Plot voronoi polygons
for _, r in cluster_geometries.iterrows():
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
for _, r in cluster_geometries.iterrows():
    sim_geo = gpd.GeoSeries(r["geometry"])
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(
        data=geo_j,
        style_function=lambda x: {"color": "black", "weight": 0.5, "fillOpacity": 0},
    )
    geo_j.add_to(m)


# %%
m.save("plymouth_clusters_map.html")

# %%
