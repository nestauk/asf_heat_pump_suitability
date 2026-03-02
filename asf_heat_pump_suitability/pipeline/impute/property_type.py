import geopandas as gpd


def impute_set_flat_properties(uprns_gdf: gpd.GeoDataFrame) -> set:
    """
    Identify domestic UPRNs which are flats from their geometries.

    Args:
        uprns_gdf (gpd.GeoDataFrame): all domestic UPRNs and their point geometries in area of interest

    Returns:
        set: UPRNs which are flats / apartments
    """
    # Count how many times each geometry occurs
    geom_counts = uprns_gdf["geometry"].value_counts()

    # Get the geometries that appear more than once (the index is the geometry)
    duplicate_geoms = geom_counts[geom_counts > 1].index

    # Filter the GeoDataFrame to only those geometries
    flats = set(uprns_gdf[uprns_gdf["geometry"].isin(duplicate_geoms)]["UPRN"])
    print(
        f"{len(flats)} flats found in UPRN dataset, N={len(uprns_gdf)}, {round(len(flats)/len(uprns_gdf)*100, 2)}% of all UPRNs"
    )
    return flats
