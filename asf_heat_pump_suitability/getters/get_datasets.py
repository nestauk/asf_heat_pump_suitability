import polars as pl
import pandas as pd
import geopandas as gpd
import os
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas
from io import StringIO
from tenacity import retry, stop_after_attempt
import warnings
import pyogrio
from typing import Tuple, List
import boto3
import json
import io

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


def load_n_hn_ashp_scores(
    nesta_hps_parquet_path: dict, read_from_s3: bool = False
) -> pd.DataFrame:
    """
    Loads the *full* Parquet containing Nesta ASHP and HN suitability scores (for *all* LAs).
    Returns a Pandas DataFrame with columns like:
      - 'lsoa'
      - 'ASHP_N_avg_score_weighted'
      - 'HN_N_avg_score_weighted'

    Args:
        nesta_hps_parquet_path (dict): Dictionary with local and s3 paths to the Nesta HP suitability Parquet file.
        read_from_s3 (bool): Whether to read the Parquet file from an s3 path or local path.

    Returns:
        pd.DataFrame
    """
    path = (
        nesta_hps_parquet_path["s3"]
        if read_from_s3
        else nesta_hps_parquet_path["local"]
    )
    logging.info(f"Loading global HP suitability data from {path}...")

    df_hps_polars = base_getters.get_pl_df_from_parquet(
        path, read_from_s3
    )  # function auto-detects "s3://" vs local
    df_hps = df_hps_polars.to_pandas()

    needed = {"lsoa", "ASHP_N_avg_score_weighted", "HN_N_avg_score_weighted"}
    if not needed.issubset(df_hps.columns):
        raise ValueError(f"Missing columns {needed} in: {df_hps.columns}")

    logging.info("Successfully loaded HN and ASHP scores.")
    return df_hps


def load_la_data(
    la_name: str,
    input_dir: str,
    s3_bucket: str,
    s3_key_dir: str,
    read_from_s3: bool = False,
) -> Tuple[pd.DataFrame, List[str], pl.DataFrame]:
    """
    Loads LA-specific data files:
      1) A Parquet containing HP suitability scores with DESNZ coverage.
      2) A JSON file listing all LSOAs for the LA.
      3) A Parquet containing average Nesta HN scores across different coverage thresholds.

    Filenames are derived from the LA name, e.g., for 'Birmingham':
      - birmingham_hp_suitability_scores_with_desnz.parquet
      - birmingham_hp_suitability_lsoas.json
      - birmingham_average_scores_by_threshold.parquet

    Args:
        la_name (str): Name of the Local Authority.
        output_dir (str): Path to the directory where the data files are stored.
        s3_bucket (str): Name of the S3 bucket where the data files are stored.
        s3_key_dir (str): Path to the directory within the S3 bucket where the data files are stored.
        read_from_s3 (bool): If True, read files from S3 using boto3. If False, read from local paths.

    Returns:
        Tuple[pd.DataFrame, List[str], pl.DataFrame]:
            - A pandas DataFrame with columns such as 'LSOA21CD', 'DESNZ_pilot_fraction', 'absolute_error'.
            - A list of LSOA codes for the LA.
            - A Polars DataFrame containing average Nesta HN scores vs DESNZ coverage thresholds.

    Raises:
        IOError: If any file fails to load, or if expected columns are missing.
    """
    hp_parquet = f"hp_suitability_scores_with_desnz/{la_name}_hp_suitability_scores_with_desnz.parquet"
    lsoa_json = f"hp_suitability_lsoas/{la_name}_hp_suitability_lsoas.json"
    avg_scores_parquet = f"avg_scores/{la_name}_average_scores_by_threshold.parquet"

    hp_parquet_local = os.path.join(input_dir, hp_parquet)
    lsoa_json_local = os.path.join(input_dir, lsoa_json)
    avg_scores_local = os.path.join(input_dir, avg_scores_parquet)

    logging.info(f"Loading data files for '{la_name}'...")

    try:
        if read_from_s3:
            # S3 reading approach
            s3_client = boto3.client("s3")

            # 1) HP Parquet
            hp_obj = s3_client.get_object(
                Bucket=s3_bucket, Key=f"{s3_key_dir}{hp_parquet}"
            )
            body = hp_obj["Body"].read()
            buffer = io.BytesIO(body)
            hp_suitability_scores_pd = pd.read_parquet(buffer)

            # 2) LSOA JSON
            lsoa_obj = s3_client.get_object(
                Bucket=s3_bucket, Key=f"{s3_key_dir}{lsoa_json}"
            )
            la_lsoas = json.loads(lsoa_obj["Body"].read())

            # 3) Average scores Parquet
            avg_hn_scores_s3_path = f"s3://{s3_bucket}/{s3_key_dir}{avg_scores_parquet}"
            average_hn_scores_coverage_df = base_getters.get_pl_df_from_parquet(
                avg_hn_scores_s3_path, read_from_s3
            )
        else:
            # Local reading approach
            hp_suitability_scores_pd = pd.read_parquet(hp_parquet_local)
            with open(lsoa_json_local, "r") as file:
                la_lsoas = json.load(file)
            average_hn_scores_coverage_df = pl.read_parquet(avg_scores_local)

        # Basic validation
        expected_cols = ["LSOA21CD", "DESNZ_pilot_fraction", "absolute_error"]
        missing_cols = [
            col for col in expected_cols if col not in hp_suitability_scores_pd.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing expected columns for {la_name}: {missing_cols}")

        return hp_suitability_scores_pd, la_lsoas, average_hn_scores_coverage_df

    except Exception as e:
        error_msg = f"Failed to load input data for {la_name} due to: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)
