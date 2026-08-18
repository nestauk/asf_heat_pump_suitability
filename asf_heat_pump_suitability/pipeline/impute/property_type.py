import geopandas as gpd
import polars as pl


def impute_set_flat_properties(
    uprns_gdf: gpd.GeoDataFrame | pl.DataFrame,
    x_col: str = "X_COORDINATE",
    y_col: str = "Y_COORDINATE",
) -> set:
    """
    Identify domestic UPRNs which are flats from their geometries or coordinates.

    Args:
        uprns_gdf (gpd.GeoDataFrame | pl.DataFrame): all domestic UPRNs. If passing a geodataframe, must contain
        UPRN point geometries in area of interest. If passing a polars dataframe, must contain X and Y coordinate columns.
        x_col (str): name of column containing X coordinate. Ignored if `uprns_gdf` is a geodataframe.
        y_col (str): name of column containing Y coordinate. Ignored if `uprns_gdf` is a geodataframe.

    Returns:
        set: UPRNs which are flats / apartments
    """
    if isinstance(uprns_gdf, gpd.GeoDataFrame):
        # Count how many times each geometry occurs
        geom_counts = uprns_gdf["geometry"].value_counts()

        # Get the geometries that appear more than once (the index is the geometry)
        duplicate_geoms = geom_counts[geom_counts > 1].index

        # Filter the GeoDataFrame to only those geometries
        flats = set(uprns_gdf[uprns_gdf["geometry"].isin(duplicate_geoms)]["UPRN"])
    else:
        flats = set(
            uprns_gdf.with_columns(
                count_geoms=pl.col("UPRN").count().over([x_col, y_col])
            ).filter(pl.col("count_geoms") > 1)["UPRN"]
        )
    print(
        f"{len(flats)} flats found in UPRN dataset, N={len(uprns_gdf)}, {round(len(flats) / len(uprns_gdf) * 100, 2)}% of all UPRNs"
    )
    return flats
