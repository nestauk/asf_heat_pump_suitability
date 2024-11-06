import pyogrio
import matplotlib.pyplot as plt

# List layers in the GeoPackage
layers = pyogrio.list_layers("heat-network-zone-map-Liverpool.gpkg")
print("Layers in GeoPackage:", layers)


# Read a specific layer into a GeoDataFrame
gdf = pyogrio.read_dataframe(
    "heat-network-zone-map-Liverpool.gpkg", layer="heat-network-zone-map-Liverpool"
)
print(gdf.head())
print(gdf.crs)
print(len(gdf))
fig, ax = plt.subplots(figsize=(12, 6))
gdf.plot(ax=ax)
plt.savefig("liverpool_heat_network_zones.png")
plt.show()
