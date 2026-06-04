import polars as pl
import geopandas as gpd
from asf_heat_pump_suitability.getters import load_geodata


def transform_df_osopen_uprn_latlon() -> pl.DataFrame:
    """
    Transform UPRN column in raw OS Open UPRN data to match format in EPC.

    Returns:
        pl.DataFrame: OS Open UPRN dataset with UPRN column in same format as in EPC
    """
    df = load_geodata.load_df_osopen_uprn()
    # Following line required to convert UPRNs to same format as in EPC
    df = df.with_columns(pl.col("UPRN").cast(pl.Float64).cast(pl.String).alias("UPRN"))

    return df


def generate_gdf_uprn_coords(
    df: pl.DataFrame,
    usecols: list = None,
    x_col: str = "X_COORDINATE",
    y_col: str = "Y_COORDINATE",
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame of British National Grid (BNG) coordinate point geometries for UPRNs from BNG x and y
    coordinates.

    Args:
        df (pl.DataFrame): dataframe with x, y coordinates in BNG (CRS: EPSG:27700) and UPRNs
        usecols (list): columns of dataframe to use. Default None to use all columns in dataframe.
        x_col (str): name of BNG x coordinate column
        y_col (str): name of BNG y coordinate column

    Returns:
        gpd.GeoDataFrame: UPRNs with BNG coordinate point geometries
    """
    print("Generating point geometries from coordinates...")
    if not usecols:
        usecols = ["*"]
    else:
        for col in [x_col, y_col]:
            if col not in usecols:
                usecols.append(col)
    df = df.select(usecols)
    df = df.to_pandas()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs="EPSG:27700",
    )

    return gdf
