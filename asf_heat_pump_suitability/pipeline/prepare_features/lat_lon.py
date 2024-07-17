import polars as pl
import geopandas as gpd
from asf_heat_pump_suitability.getters import get_datasets


def transform_df_osopen_uprn_latlon() -> pl.DataFrame:
    """
    Transform UPRN column in raw OS Open UPRN data to match format in EPC.

    Args:
        df (pl.DataFrame): OS Open UPRN dataset

    Returns:
        pl.DataFrame: OS Open UPRN dataset with UPRN column in same format as in EPC
    """
    df = get_datasets.get_df_osopen_uprn_latlon()
    # Following line required to convert UPRNs to same format as in EPC
    df = df.with_columns(pl.col("UPRN").cast(pl.Float64).cast(pl.String).alias("UPRN"))

    return df


def generate_gdf_uprn_coords(df: pl.DataFrame) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame of British National Grid (BNG) coordinate point geometries for UPRNs from BNG x and y
    coordinates.

    Args:
        df (pl.DataFrame): dataframe with x, y coordinates and UPRNs

    Returns:
        gpd.GeoDataFrame: UPRNs with BNG coordinate point geometries
    """
    df = df.select(["UPRN", "X_COORDINATE", "Y_COORDINATE"]).to_pandas()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.X_COORDINATE, df.Y_COORDINATE),
        crs="EPSG:27700",
    )
    gdf = gdf[["UPRN", "geometry"]]

    return gdf
