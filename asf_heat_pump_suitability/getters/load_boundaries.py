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
        return la_boundaries_gdf[
            la_boundaries_gdf["LAD23NM"].str.contains(select_las.title())
        ]
    else:
        print(f"Loading Local Authority boundaries for {select_las}...")
        select_las = [la.title() for la in select_las]
        la_boundaries_gdf = la_boundaries_gdf[
            la_boundaries_gdf["LAD23NM"].str.contains("|".join(select_las))
        ]
        matches = [
            la
            for la in select_las
            if any(la_boundaries_gdf["LAD23NM"].str.contains(la))
        ]

        # Raise exception if any boundaries are not found for LAs in select_las
        if len(set(select_las).difference(matches)) > 0:
            raise Exception(
                f"Could not find boundaries for the following Local Authorities: {set(select_las).difference(matches)}"
            )
        else:
            return la_boundaries_gdf
