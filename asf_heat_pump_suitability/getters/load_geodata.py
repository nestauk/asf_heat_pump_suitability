import polars as pl
import geopandas as gpd
import os
import pandas as pd
from typing import Optional, List
import s3fs

from osbng import grids

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def load_df_osopen_uprn(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid X and Y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OSOpen UPRNs...")
    path = config["data"]["geodata"]["uk_osopen_uprn"]
    filename = os.path.basename(path).split("_csv")[0]
    df = base_getters.get_df_from_zip_csv_s3(
        path,
        extract_file=f"{filename}.csv",
        **kwargs,
    )

    return df


def load_gdf_bng_grid_squares() -> gpd.GeoDataFrame:
    """
    Load British National Grid squares at 100km resolution, CRS 27700.

    Returns:
        gpd.GeoDataFrame: British National Grid square codes and their corresponding polygons
    """
    return gpd.GeoDataFrame.from_features(grids.bng_grid_100km, crs=27700)


def load_gdf_heat_network_zones(local_authority: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with heat network zone polygons in given Local Authority.

    Args:
        local_authority (str): Local Authority or Local Authorities to load Heat Network zone polygons for.
        e.g. `plymouth` for Plymouth Local Authority; `greater_manchester_las` for Greater Manchester Combined Authority (all 10 LAs in Greater Manchester).
        See config/base.yaml under the `constant` key for options.

        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: polygons of heat network zones in given Local Authority.
    """

    # TODO: this will currently only work for HN zone files defined in config/base.yaml.
    # We need to change this to make it work for all other HN zone files,
    # for example by concatenating all HN zone files and checking if the geometries intersect with the local authority boundary.

    gdf = gpd.GeoDataFrame()

    # Local authority (e.g. `plymouth`) or group of local authorities (e.g. `greater_manchester_las`)
    local_authority = local_authority.lower()

    # Load heat network zone geodata for the specified local authority or group of local authorities.
    try:
        gdf = base_getters.get_gdf_from_gpkg_s3_path(
            path=config["data"]["geodata"]["heat_network_zones"][local_authority],
            **kwargs,
        )
        # Assume first column with `ID` substring is the zone ID column
        # Note original ID column retained in case of erroneous ID assignment
        gdf = _extend_gdf_hn_zone_id(gdf)

    except (ValueError, KeyError):
        # Get list of LAs (e.g. for `greater_manchester_las` this means getting a list of all individual LAs) to attempt
        # loading heat network zone geodata for each LA individually if no geodata found for the whole group of LAs.
        list_las = config["constant"][local_authority]["la_names"]
        list_las = list_las if isinstance(list_las, list) else [list_las]

        # If gdf is still empty and `local_authority` represents a group of LAs
        if gdf.empty and len(list_las) > 1:
            # Check if heat network zone geodata is available for each LA in the list, and if so, load it and concatenate
            # it to a single geodataframe.
            for la in list_las:
                try:
                    gdf = pd.concat(
                        [
                            gdf,
                            base_getters.get_gdf_from_gpkg_s3_path(
                                path=config["data"]["geodata"]["heat_network_zones"][
                                    la
                                ],
                                **kwargs,
                            ),
                        ],
                        ignore_index=True,
                    )
                    # Deal with different ID column names in different geodataframes by renaming the ID column to "ZoneID"
                    gdf = _extend_gdf_hn_zone_id(gdf)
                except (ValueError, KeyError):
                    print(
                        f"No heat network zone geodata found for Local Authority: {la}."
                    )
    finally:
        if len(gdf) > 0:
            gdf.set_geometry("geometry", inplace=True)
            print(
                f"Heat network zone geodataframe successfully loaded for {local_authority} with CRS {gdf.crs}."
            )
    return gdf


def _extend_gdf_hn_zone_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add `HNZoneID` column to heat network geodataframe using existing ID columns.

    Args:
        gdf (gpd.GeoDataFrame): heat network zones

    Returns:
        gpd.GeoDataFrame: heat network zones with `HNZoneID` column
    """
    id_cols = [col for col in gdf.columns if "ID" in col]
    id_col = id_cols[0]
    print(f"Using Heat Network Zone {id_col} column as ID. Other options: {id_cols}")

    gdf["HNZoneID"] = gdf[id_col]
    gdf = gdf.rename(columns={id_col: f"original_{id_col}"})
    return gdf


def load_gdf_spatial_signatures_gb(
    detail_level: str = "full", **kwargs
) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with polygons in GB from the Spatial Signatures Framework  classified by their
    Spatial Signature type. CRS British National Grid (27700). (Source: https://doi.org/10.6084/m9.figshare.16691575).

    Two versions of the dataset are available, differing in the geometric detail of the
    polygon geometries:

    - "simplified":
        Polygon geometries have been geometrically simplified, containing fewer coordinate
        vertices than the full-resolution version.
        Attribute fields include: "id" (int64) and "type" (str).

    - "full":
        Polygon geometries are provided at full resolution, retaining the complete set of
        coordinate vertices.
        Attribute fields include: "id" (str), "code" (str), and "type" (str).

    Both versions contain 96,704 polygons.

    Args:
        detail_level (str, optional): Which level of descriptive detail to load.
            Must be either "simplified" or "full". Defaults to "simplified".

    Returns:
        gpd.GeoDataFrame: spatial signature polygons in GB.
    """

    if detail_level not in {"full", "simplified"}:
        raise ValueError(
            f"detail_level must be 'full' or 'simplified', not {detail_level}"
        )

    print("Loading spatial signatures dataset...")
    gdf = base_getters.get_gdf_from_gpkg_s3_path(
        path=config["data"]["geodata"]["gb_spatial_signatures"][detail_level],
        **kwargs,
    )

    print(
        f"Spatial signatures {detail_level} geodataframe successfully loaded with CRS {gdf.crs}."
    )

    return gdf


def load_gdf_os_openmap_layer(
    layer: str, grid_squares: Optional[List[str]] = None, **kwargs
) -> gpd.GeoDataFrame:
    """
    Load specified OS OpenMap Local or Greenspace layer for Great Britain or optionally for a specific grid square.
    CRS British National Grid (27700).

    Find full list of green space sites here: https://docs.os.uk/os-downloads/products/land-and-terrain-portfolio/os-open-greenspace/os-open-greenspace-technical-specification/code-lists/functionvalue#code-list-functionvalue

    Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf

    Args:
        layer (str): name of layer to load. See layer options below.
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded. Default None to load whole GB.
        **kwargs for geopandas.read_file()

    Layer options:
        'building',
        'car_charging_point',
        'electricity_transmission_line',
        'foreshore',
        'functional_site',
        'glasshouse',
        'greenspace_site',
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
    if not grid_squares:  # Load all of GB
        if layer == "greenspace_site":
            raise ValueError(
                "Greenspace site not implemented for GB yet. Please select a grid square."
            )

        print(f"Loading OS OpenMap Local - {layer.title()}...")
        return gpd.read_file(
            filename=config["data"]["geodata"]["gb_os_openmap_local"],
            layer=layer,
            **kwargs,
        ).drop_duplicates(subset="ID")

    else:
        if not isinstance(grid_squares, List):
            grid_squares = [grid_squares]

        # Reformat layer name to how it appears in file name
        if layer in ["surface_water_area", "surface_water_line"]:
            layer = "_".join(["SurfaceWater", layer[-4:].title()])
        else:
            layer = layer.replace("_", " ").title().replace(" ", "")

        if layer == "GreenspaceSite":
            file_path = config["data"]["geodata"]["grid_square_os_openmap_greenspace"]
        else:
            file_path = config["data"]["geodata"]["grid_square_os_openmap_local"]

        files = [file_path.format(square=code, layer=layer) for code in grid_squares]

        gdfs = []

        for file in files:
            print(f"\nLoading OS OpenMap layer - {layer.title()} file: {file}")
            gdfs.append(gpd.read_file(file, **kwargs))

        gdf = pd.concat(gdfs)
        id_col = "ID" if "ID" in gdf.columns else "id"

        return gdf.drop_duplicates(subset=[id_col, "geometry"])


def load_gdf_os_openroad(
    grid_squares: Optional[List[str]] = None, **kwargs
) -> gpd.GeoDataFrame:
    """
    Load road link data from OS OpenRoad for Great Britain or optionally for a specific grid square (or list of grid squares). CRS British National Grid (27700).
    Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded. Default None to load whole GB.
        **kwargs for geopandas.read_file()
    Returns:
        gpd.GeoDataFrame: OS OpenRoad linestrings for specified grid squares.
    """
    if not grid_squares:
        fs = s3fs.S3FileSystem()
        file_path = config["data"]["geodata"]["gb_os_openroad"]
        files = fs.glob(f"{file_path}*_RoadLink.shp")
        gdfs = []
        for file in files:
            print(f"\nLoading OS OpenRoad file: {file}")
            gdfs.append(gpd.read_file(f"s3://{file}"))

        gdf = pd.concat(gdfs)
    else:
        file_path = config["data"]["geodata"]["grid_square_os_openroad"]
        if not isinstance(grid_squares, List):
            grid_squares = [grid_squares]
        files = [file_path.format(square=code) for code in grid_squares]

        gdfs = []

        for file in files:
            print(f"\nLoading OS OpenRoad file: {file}")
            gdfs.append(gpd.read_file(file))

        gdf = pd.concat(gdfs)

    return gdf


def load_gdf_poi() -> gpd.GeoDataFrame:
    """
    Load and process Points of Interest data. CRS EPSG 4326.

    Returns:
        gpd.GeoDataFrame: Processed POI data containing types of POI specified

    Raises:
        ValueError: If required columns are missing
    """
    print("Loading POI data...")

    required_columns = [
        "id",
        "country",
        "main_category",
        "alternate_category",
        "geometry",
    ]
    poi = gpd.read_file(
        filename=config["data"]["geodata"]["UK_poi_locations"],
        columns=required_columns,
        layer="poi_uk",
    ).to_crs("EPSG:4326")
    print(f"POI CRS: {poi.crs}")

    return poi
