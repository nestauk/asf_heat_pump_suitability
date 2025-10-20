import geopandas as gpd
import pandas as pd
from typing import Optional, List
from asf_heat_pump_suitability import config


def load_gdf_os_openmap_local_layer(
    layer: str, grid_squares: Optional[List[str]] = None, **kwargs
) -> gpd.GeoDataFrame:
    """
    Load specified OS OpenMap Local layer. CRS British National Grid (27700).

    Args:
        layer (str): name of layer to load.
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded.
        Default None to load whole GB.

    Layer options:
        'building',
        'car_charging_point',
        'electricity_transmission_line',
        'foreshore',
        'functional_site',
        'glasshouse',
        'important_building',
        'motorway_junction',
        'named_place',
        'railway_station',
        'railway_track',
        'railway_tunnel',
        'road',
        'road_tunnel',
        'roundabout',
        'surface_water_area',
        'surface_water_line',
        'tidal_boundary',
        'tidal_water',
        'woodland'

    Returns:
        gpd.GeoDataFrame: OS OpenMap Local geometries for specified layer
    """
    if not grid_squares:
        print(f"\nLoading OS OpenMap Local - {layer.title()}...")
        return gpd.read_file(
            config["data"]["geodata"]["gb_os_openmap_local"], layer=layer, **kwargs
        )

    else:
        layer = layer.replace("_", " ").title().replace(" ", "")
        file_path = config["data"]["geodata"]["grid_square_os_openmap_local"]
        files = [file_path.format(square=code, layer=layer) for code in grid_squares]

        gdfs = []

        for file in files:
            print(f"\nLoading OS OpenMap Local - {layer.title()} file: {file}")
            gdfs.append(gpd.read_file(file, **kwargs))

        return pd.concat(gdfs)
