"""
Functions to load raw census geography boundaries, like Local Authority; ward; output areas, etc. No/minimal preprocessing occurs in these functions.
"""

import logging
from typing import List

import geopandas as gpd

from asf_heat_pump_suitability import config

logger = logging.getLogger(__name__)


def load_gdf_local_authority_boundaries(
    select_las: str | List[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load boundaries for specified Local Authority Districts. CRS British National Grid (27700).

    Args:
        select_las (str | List[str]): selected Local Authorities to load boundaries for. Optional. Default None to load all UK
        Local Authority boundaries.

    Returns:
        gpd.GeoDataFrame: boundaries for specified Local Authority Districts
    """
    la_boundaries_gdf = gpd.read_file(config["data"]["geodata"]["boundaries"]["UK_ons_lad_bounds"])

    if not select_las:
        logger.info("Loading Local Authority boundaries for UK...")
        return la_boundaries_gdf
    elif isinstance(select_las, str):
        logger.info(f"Loading Local Authority boundaries for {select_las}...")
        la_boundaries_gdf = la_boundaries_gdf[la_boundaries_gdf["LAD23NM"].str.contains(select_las.title())]
        # Raise exception if boundaries are not found for LA specified by select_las
        if len(la_boundaries_gdf) == 0:
            raise Exception(f"Could not find boundaries for the following Local Authority: {select_las}")
        else:
            return la_boundaries_gdf
    else:
        logger.info(f"Loading Local Authority boundaries for {select_las}...")
        la_boundaries_gdf = la_boundaries_gdf[
            # Filter to exact LA name matches, case insensitive
            la_boundaries_gdf["LAD23NM"].str.fullmatch("|".join(select_las), case=False)
        ]
        matches = set(la_boundaries_gdf["LAD23NM"].unique())

        # Raise exception if any boundaries are not found for LAs in select_las
        missing_las = set(select_las).difference(matches)
        if len(missing_las) > 0:
            raise Exception(f"Could not find boundaries for the following Local Authorities: {missing_las}")
        else:
            return la_boundaries_gdf
