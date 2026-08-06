"""
Functions to load raw census geography boundaries, like Local Authority; ward; output areas, etc. No/minimal preprocessing occurs in these functions.
"""

import pandas as pd
import geopandas as gpd
import shapely
from typing import List

from asf_heat_pump_suitability import config


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
    la_boundaries_gdf = gpd.read_file(
        config["data"]["geodata"]["boundaries"]["UK_ons_lad_bounds"]
    )

    if not select_las:
        print("Loading Local Authority boundaries for UK...")
        return la_boundaries_gdf
    elif isinstance(select_las, str):
        print(f"Loading Local Authority boundaries for {select_las}...")
        la_boundaries_gdf = la_boundaries_gdf[
            la_boundaries_gdf["LAD23NM"].str.lower().str.contains(select_las.lower())
        ]
        # Raise exception if boundaries are not found for LA specified by select_las
        if len(la_boundaries_gdf) == 0:
            raise Exception(
                f"Could not find boundaries for the following Local Authority: {select_las}"
            )
        else:
            return la_boundaries_gdf
    else:
        print(f"Loading Local Authority boundaries for {select_las}...")
        la_boundaries_gdf = la_boundaries_gdf[
            # Filter to exact LA name matches, case insensitive
            la_boundaries_gdf["LAD23NM"].str.fullmatch("|".join(select_las), case=False)
        ]
        matches = set(la_boundaries_gdf["LAD23NM"].unique())

        # Raise exception if any boundaries are not found for LAs in select_las
        missing_las = set(select_las).difference(matches)
        if len(missing_las) > 0:
            raise Exception(
                f"Could not find boundaries for the following Local Authorities: {missing_las}"
            )
        else:
            return la_boundaries_gdf


def load_gdf_ward_boundaries(
    select_las: str | List[str] = None, la_boundaries_gdf: gpd.GeoDataFrame = None
) -> gpd.GeoDataFrame:
    """
    Load boundaries for the whole of the UK.
    If select_las and la_boundaries_gdf are specified, load boundaries for the specified Local Authority Districts.
    CRS British National Grid (27700).

    Args:
        select_las (str | List[str]): selected Local Authorities to load boundaries for. Optional. Default None to load all.
        la_boundaries_gdf (gpd.GeoDataFrame): boundaries for specified Local Authority Districts. Optional. Default None to load all.

    Returns:
        gpd.GeoDataFrame: boundaries for specified Local Authority Districts or all UK if no selection is made.
    """

    wards_gdf = pd.read_parquet(
        config["data"]["geodata"]["boundaries"]["UK_ward_boundaries"]
    )
    wards_gdf = gpd.GeoDataFrame(
        wards_gdf, geometry=gpd.GeoSeries.from_wkt(wards_gdf.geometry), crs="EPSG:4326"
    ).to_crs(epsg=27700)

    if select_las:
        if la_boundaries_gdf is None:
            raise ValueError(
                "la_boundaries_gdf must be provided if select_las is specified."
            )
        print(f"Loading ward boundaries for {select_las}...")
        wards_gdf = wards_gdf.sjoin(
           la_boundaries_gdf, how="inner", predicate="intersects"
        )
        return wards_gdf
    else:
        print("Loading ward boundaries for the whole of the UK...")
        return wards_gdf
