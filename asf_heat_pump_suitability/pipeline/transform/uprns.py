"""Functions to transform UPRN data."""

import logging

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability.getters.get_epc import load_set_valid_epc_uprns
from asf_heat_pump_suitability.utils import geo_utils

logger = logging.getLogger(__name__)


def generate_gdf_uprn_coords(
    df: pl.DataFrame,
    usecols: list = None,
    x_col: str = "X_COORDINATE",
    y_col: str = "Y_COORDINATE",
) -> gpd.GeoDataFrame:
    """Generate GeoDataFrame of British National Grid (BNG) coordinate point geometries for UPRNs.

    Args:
        df (pl.DataFrame): dataframe with x, y coordinates in BNG (CRS: EPSG:27700) and UPRNs
        usecols (list): columns of dataframe to use. Default None (all columns).
        x_col (str): name of BNG x coordinate column
        y_col (str): name of BNG y coordinate column

    Returns:
        gpd.GeoDataFrame: UPRNs with BNG coordinate point geometries
    """
    if not usecols:
        usecols = ["*"]
    else:
        for col in [x_col, y_col]:
            if col not in usecols:
                usecols.append(col)
    df = df.select(usecols).to_pandas()

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs="EPSG:27700",
    )


def filter_gdf_residential_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter UPRNs to residential-only by retaining UPRNs that appear in the domestic EPC
    register, OR are located within a building footprint AND are not in the commercial EPC
    register and/or a building type unlikely to contain residential properties.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be filtered
        buildings_gdf (gpd.GeoDataFrame): all building footprints in area of interest
        non_residential_buildings_gdf (gpd.GeoDataFrame): polygons of buildings unlikely to
            contain residential properties

    Returns:
        gpd.GeoDataFrame: UPRNs assumed to represent residential properties
    """
    print("Filtering to residential UPRNs...")
    non_residential_uprns = set(uprn_gdf.sjoin(non_residential_buildings_gdf, how="inner", predicate="within")["UPRN"])
    non_residential_uprns.update(load_set_valid_epc_uprns(epc_type="commercial"))

    uprns_in_buildings = set(uprn_gdf.sjoin(buildings_gdf, how="inner", predicate="intersects")["UPRN"])
    epc_residential_uprns = load_set_valid_epc_uprns(epc_type="domestic")

    return uprn_gdf[
        (~uprn_gdf["UPRN"].isin(non_residential_uprns) & uprn_gdf["UPRN"].isin(uprns_in_buildings))
        | uprn_gdf["UPRN"].isin(epc_residential_uprns)
    ]


def map_dict_uprns_to_building_id(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    id_col: str,
    predicate: str = "intersects",
) -> dict:
    """Create a mapping of UPRNs (keys) to the building ID (values) of the building they
    intersect with.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs with geospatial point data
        buildings_gdf (gpd.GeoDataFrame): building footprints
        id_col (str): name of building ID column in `buildings_gdf`
        predicate (str): spatial join predicate — "intersects" or "within". Default "intersects".

    Returns:
        dict: mapping of UPRNs to building IDs
    """
    uprns_gdf = geo_utils.verify_gdf_crs(uprns_gdf)
    buildings_gdf = geo_utils.verify_gdf_crs(buildings_gdf)
    return uprns_gdf.sjoin(buildings_gdf, how="inner", predicate=predicate).set_index("UPRN").to_dict()[id_col]
