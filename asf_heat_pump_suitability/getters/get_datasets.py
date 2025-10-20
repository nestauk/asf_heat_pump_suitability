import polars as pl
import pandas as pd
import geopandas as gpd
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas
from io import StringIO
from tenacity import retry, stop_after_attempt
import warnings
from typing import Tuple

# Ignore RunTimeWarning when loading Microsoft building footprint files
# as reading from gzipped stream should be faster than unzipping and loading data
warnings.filterwarnings("ignore", category=RuntimeWarning, message="VSIFSeekL")


def get_df_ons_pd(**kwargs) -> pl.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: postcode directory for Great Britain
    """
    df = base_getters.get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        schema=schemas.onspd_schema,
        **kwargs,
    )

    return df


def load_gdf_ons_council_bounds(**kwargs) -> gpd.GeoDataFrame:
    """
    Load ONS council bounding polygons for the UK (CRS: EPSG:27700).

    Args:
        **kwargs for gpd.read_file

    Returns:
        gpd.GeoDataFrame: ONS councils with bounding polygons
    """
    gdf = gpd.read_file(config["data_source"]["UK_ons_lad_bounds"], **kwargs)

    return gdf


def load_df_microsoft_building_footprint_links() -> pd.DataFrame:
    """
    Load Microsoft Global ML Building Footprints data links file containing URLs to all building footprint files
    available.

    Returns:
        pd.DataFrame: Microsoft Global ML Building Footprints data links
    """
    logging.info("Loading Microsoft building footprint data-links file")
    df = pd.read_csv(
        config["data_source"]["global_microsoft_building_footprint_links"],
        dtype=schemas.microsoft_datalinks,
    )

    return df


def load_gdf_microsoft_building_footprints(url: str) -> gpd.GeoDataFrame:
    """
    Load Microsoft building footprints file (CRS: EPSG:4326).

    Args:
        url (str): URL to Microsoft building footprint file

    Returns:
        gpd.GeoDataFrame: Microsoft building footprint polygons
    """
    gdf = gpd.read_file(
        f"GeoJSONSeq:/vsigzip//vsicurl/{url}", engine="pyogrio", use_arrow=True
    )

    return gdf


@retry(stop=stop_after_attempt(4))
def load_gdf_inspire_land_parcels(path: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Load land registry's index polygons spatial data (INSPIRE) showing the geometry and extent of registered freehold
    properties in England and Wales. CRS EPSG:27700, British National Grid.

    Args:
        path (str): path to INSPIRE land parcel file
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: registered land extent polygons for one council
    """
    logging.info(f"Loading INSPIRE land parcel file: {path}")
    gdf = gpd.read_file(path, engine="pyogrio", **kwargs)

    return gdf


def get_df_ons_garden_space_avg(**kwargs) -> pl.DataFrame:
    """
    Get raw ONS 'Access to garden space, Great Britain' dataset.

    Args:
        **kwargs for pl.read_excel

    Returns:
        pl.DataFrame: raw ONS 'Access to garden space' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["GB_ons_garden_space_access"]
    )
    df = pl.read_excel(
        content,
        sheet_name="MSOA gardens",
        engine="calamine",
        **kwargs,
    )
    return df


def get_df_osopen_uprn_latlon(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid x and y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs fo pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OS OpenMap UPRN dataset...")
    df = base_getters.get_df_from_zip_csv_s3(
        config["data_source"]["GB_osopen_uprn_latlon"],
        extract_file="osopenuprn_202405.csv",
        **kwargs,
    )

    return df


def load_gdf_historic_england_conservation_areas(**kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with building conservation area polygons from Historic England (CRS: EPSG:4326).

    Args:
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: polygons of building conservation areas in England
    """
    gdf = gpd.read_file(
        config["data_source"]["E_historic_england_conservation_areas"], **kwargs
    )

    return gdf


def load_gdf_welsh_gov_conservation_areas(**kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with building conservation area polygons from the Welsh Government (CRS: EPSG:27700 British
    National Grid).

    Args:
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: polygons of building conservation areas in Wales
    """
    gdf = gpd.read_file(
        config["data_source"]["W_welsh_gov_conservation_areas"], **kwargs
    )

    return gdf


def get_df_ons_number_of_households() -> pl.DataFrame:
    """
    Get raw ONS 'Number of households' per LSOA for England and Wales.

    Returns:
        pl.DataFrame: raw ONS 'Number of households' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_number_of_households"],
    )
    content_str = content.decode("utf-8")  # convert bytes to string
    content_file = StringIO(content_str)  # convert string to file-like object
    df = pl.read_csv(content_file, skip_rows=6, has_header=True)
    # Preprocessing steps due to white space
    # Remove the last eight rows
    df = df.slice(0, len(df) - 9)
    # Remove the first row
    df = df.slice(1, len(df) - 1)
    return df


def get_df_ons_land_area() -> pl.DataFrame:
    """
    Get raw ONS 'land area' dataset. Contains Standard Area Measurements of ‘Land Area’ (Area to Mean High Water
    Excluding Area of Inland Water) for England and Wales.

    Returns:
        pl.DataFrame: raw ONS 'land area' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["EW_census_land_area"],
    )
    content_str = content.decode("utf-8")  # convert bytes to string
    content_file = StringIO(content_str)  # convert string to file-like object

    # dtypes specificed as polars read csv was inferring wrong data types and throwing error
    dtypes = {
        "LSOA21CD": pl.Utf8,
        "LSOA21NM": pl.Utf8,
        "Extent of the Realm (Area in KM2)": pl.Float64,
        "Clipped to the Coastline (Area in KM2)": pl.Float64,
        "Area of Inland Water (KM2)": pl.Float64,
        "Land Count (Area in KM2)": pl.Float64,
        "LTLA22CD": pl.Utf8,
        "LTLA22NM": pl.Utf8,
        "LTLA22NMW": pl.Utf8,
    }
    df = pl.read_csv(content_file, dtypes=dtypes, has_header=True)
    return df


def get_df_spa_offgasgrid() -> pl.DataFrame:
    """
    Get off gas grid data from Supply Point Administration dataset

    Returns:
        pl.DataFrame: raw off gas grid dataset
    """
    df = base_getters.get_df_from_excel_s3_path(
        config["data_source"]["UK_spa_offgasgrid"], sheet_name="Off-Gas Postcodes 2024"
    )
    return df


def load_gdf_listed_buildings(nation: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Get raw Listed Buildings polygons dataset for specified nation. CRS EPSG:27700, British National Grid.

    Args:
        nation (str): nation to load listed buildings data for. Options: "England"; "Scotland", "Wales".
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: raw Listed Buildings dataset for specified nation
    """
    if nation.lower() == "england":
        gdf = gpd.read_file(
            config["data_source"]["E_historicengland_listed_buildings"], **kwargs
        )
    elif nation.lower() == "wales":
        gdf = gpd.read_file(config["data_source"]["W_cadw_listed_buildings"], **kwargs)
    elif nation.lower() == "scotland":
        gdf = gpd.read_file(
            config["data_source"]["S_scottish_gov_listed_buildings"], **kwargs
        )
    else:
        raise ValueError(
            "Please set `nation` to either 'England', 'Scotland', or 'Wales'."
        )
    return gdf


def load_gdf_ons_lsoa_bounds(**kwargs) -> gpd.GeoDataFrame:
    """
    Load raw 2021 LSOA geospatial boundary polygons for England and Wales from ONS. CRS
    British National Grid (EPSG:27700).

    Args:
        **kwargs for geopandas.read_file()

    Returns:
        gpd.GeoDataFrame: boundary polygons for 2021 LSOAs
    """
    return gpd.read_file(config["data_source"]["EW_lsoa_bounds"], **kwargs)


def load_gdf_scotgov_data_zone_bounds(**kwargs) -> gpd.GeoDataFrame:
    """
    Load raw 2011 Data Zone geospatial boundary polygons and area data for Scotland from the Scottish Government. CRS
    British National Grid (EPSG:27700).

    Args:
        **kwargs for geopandas.read_file()

    Returns:
        gpd.GeoDataFrame: boundary polygons and area standard area measurement data for 2011 Scottish Data Zones
    """
    return gpd.read_file(
        config["data_source"]["S_scottish_gov_DZ2011_boundaries"], **kwargs
    )


def load_df_nrs_dwellings() -> pl.DataFrame:
    """
    Load 2023 dwelling counts per 2011 Data Zone in Scotland from National Records of Scotland. Data remains in raw
    form with light processing to correct column headers and dtypes.

    Returns:
        pl.DataFrame: dwelling counts per 2011 Scottish Data Zone
    """
    df = base_getters.get_df_from_excel_s3_path(
        config["data_source"]["S_NRScotland_households"], sheet_name="2023"
    )
    # Remove empty rows and set column headers to correct names
    df.columns = df.row(2)
    df = df[3:].cast(schemas.nrs_dwellings)

    return df


def load_desnz_geodata(
    gpkg_path: str, shp_path: str, layer_name: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load DESNZ heat network polygons from a GeoPackage and LSOA shapefile.

    Args:
        desnz_hn_gpkg_path (str): Path to the DESNZ Heat Network GeoPackage.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        layer_name (str): Layer name in the GeoPackage.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            - DESNZ heat network zones as a GeoDataFrame.
            - LSOA polygons as a GeoDataFrame.
    """
    hn_gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name)
    lsoa_gdf = gpd.read_file(shp_path)
    return hn_gdf, lsoa_gdf


def load_df_scot_gov_data_zone_LA() -> pd.DataFrame:
    """
    Load Scottish Gov data zones with the local authority they are part of.
    """
    df = pd.read_csv(
        config["data_source"]["S_data_zone_LA"],
        usecols=["DZ22_Code", "DZ22_Name", "LA_Name", "LA_Code", "SPD_Name"],
        encoding="iso-8859-1",
    )
    return df


def load_df_gov_LSOA_LA() -> pd.DataFrame:
    """
    Load data.gov data of LSOA and the local authority they are part of.
    """
    df = pd.read_csv(
        config["data_source"]["EW_LSOA_LA"], usecols=["LSOA21CD", "LAD23NM", "LAD23CD"]
    )
    return df


def load_df_gov_LSOA_region() -> pd.DataFrame:
    """
    Load data.gov data of LSOA and the region they are part of.
    """
    df = pd.read_csv(
        config["data_source"]["EW_LSOA_region"], usecols=["LSOA21CD", "RGN22NM"]
    )


def load_df_lsoa_lad_lookup(**kwargs) -> pl.DataFrame:
    """
    Load LSOA to LAD lookup table from ONS.

    Args:
        **kwargs for `polars.read_csv()`

    Returns:
        pl.DataFrame: LSOA to LAD lookup table for England and Wales
    """
    df = pl.read_csv(config["data_source"]["EW_ons_lsoa_lad_lookup"], **kwargs)
    return df


def load_df_dz_lookup(**kwargs) -> pl.DataFrame:
    """
    Load Data Zone to LAD lookup table from Scottish Government.

    Args:
        **kwargs for `polars.read_csv()`

    Returns:
        pl.DataFrame: DZ to LAD lookup table for Scotland
    """
    df = pl.read_csv(
        config["data_source"]["S_dz_lookup"],
        infer_schema_length=5000,
        ignore_errors=True,
        **kwargs,
    )
    return df
