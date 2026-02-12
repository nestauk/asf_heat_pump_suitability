"""
Functions to transform UPRN data.

Contains script to filter OS UPRNs to residential UPRNs only. UPRNs that meet any of the following criteria are assumed to be
residential:
- UPRNs geolocated inside a building footprint but not in a 'non-residential' building type (see non_residential_entities.py) and not in the non-domestic EPC register
- UPRNs found in the domestic EPC register

To run the script:
python asf_heat_pump_suitability/pipeline/transform/uprns.py

Set the optional `local_authorities` parameter to `plymouth`, `plymouth_similar`, or `sampling_areas`.
Set to `plymouth` to run for Plymouth Local Authority; `plymouth_similar` to run for Plymouth plus four other similar
Local Authorities (Liverpool, Portsmouth, Southampton, Swansea); 'sampling_areas' to run for Plymouth plus five other
Local Authorities for sampling buildings (Bath, Bradford, Glasgow, Manchester, Nottingham); or do not use to run for all
of Great Britain.
"""

import geopandas as gpd
import polars as pl
import logging
import argparse
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import geo_utils


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
        usecols (list): columns of dataframe to use. Default None.
        x_col (str): name of BNG x coordinate column
        y_col (str): name of BNG y coordinate column

    Returns:
        gpd.GeoDataFrame: UPRNs with BNG coordinate point geometries
    """
    # If usecols is not specified, use all columns in the dataframe
    if not usecols:
        usecols = ["*"]
    else:
        # If usecols is specified, check that X and Y coordinate columns are included, otherwise add them
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


def load_set_valid_epc_uprns(epc_type: str) -> set:
    """
    Load set of valid EPC UPRNs from either commercial or domestic EPC registers.

    Args:
        epc_type (str): {"commercial", "domestic"} the type of EPC to load valid UPRNs from

    Returns:
        set: valid UPRNs from specified EPC dataset
    """
    print(f"Loading UPRNs from {epc_type} EPC register...")
    df = base_getters.load_df_from_s3(config["data"]["epc"][epc_type], columns="UPRN")
    before = len(df)
    df = df.with_columns(
        # Remove any invalid UPRNs (i.e. those IDs which are generated in EPC preprocessing generated from concatenating building ref number and address)
        # These are not true UPRNs that can be used in joins across other datasets
        pl.col("UPRN")
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64)
        .alias("UPRN")
    ).drop_nulls()
    logging.info(
        f"{before - len(df)} invalid UPRNs dropped from {epc_type} EPC register. {len(df)} valid UPRNs remaining"
    )

    return set(df["UPRN"])


def filter_gdf_residential_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Filter UPRNs to residential UPRNs only by retaining UPRNs which appear in domestic EPC register, OR are located within
    a building footprint AND are not in the commercial EPC register and / or a building type that is unlikely to contain
    residential properties, e.g. hospital, train station, museum etc.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be filtered
        building_gdf (gpd.GeoDataFrame): all building footprints in area of interest
        non_residential_buildings_gdf (gpd.GeoDataFrame): polygons of buildings which are unlikely to contain residential
        properties

    Returns:
        gpd.GeoDataFrame: UPRNs which are assumed to represent residential properties with their point geometries
    """
    print("Filtering to residential UPRNs...")
    # Find UPRNs which are in the non-residential buildings
    non_residential_uprns = set(
        uprn_gdf.sjoin(non_residential_buildings_gdf, how="inner", predicate="within")[
            "UPRN"
        ]
    )

    # Get valid non-residential EPC UPRNs
    non_residential_uprns.update(load_set_valid_epc_uprns(epc_type="commercial"))

    # Find UPRNs which are in any building (i.e. remove UPRNs which represent outdoor addressable locations)
    uprns_in_buildings = set(
        uprn_gdf.sjoin(buildings_gdf, how="inner", predicate="intersects")["UPRN"]
    )

    # Get valid residential UPRNs
    epc_residential_uprns = load_set_valid_epc_uprns(epc_type="domestic")

    return uprn_gdf[
        (
            # Filter to UPRNs which are in buildings AND not in non-residential UPRNs list
            (~uprn_gdf["UPRN"].isin(non_residential_uprns))
            & (uprn_gdf["UPRN"].isin(uprns_in_buildings))
        )
        # Or UPRNs which are in domestic EPC register
        | (uprn_gdf["UPRN"].isin(epc_residential_uprns))
    ]


def map_dict_uprns_to_building_id(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    id_col: str,
    predicate: str = "intersects",
) -> dict:
    """
    Create a mapping of UPRNs (keys) to the building ID (values) of the building they are located within or intersect with.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs with geospatial point data
        buildings_gdf (gpd.GeoDataFrame): building footprints
        id_col (str): name of building ID column in `buildings_gdf`
        predicate (str): how to join buildings and UPRNs, of `intersects` which joins UPRNs with building footprints
        they intersect with, or `within` which joins UPRNs to building footprints they are located within. Default `intersects`.

    Returns:
        dict: mapping of UPRNs to building IDs
    """
    uprns_gdf = geo_utils.verify_gdf_crs(uprns_gdf)
    buildings_gdf = geo_utils.verify_gdf_crs(buildings_gdf)

    return (
        uprns_gdf.sjoin(buildings_gdf, how="inner", predicate=predicate)
        .set_index("UPRN")
        .to_dict()[id_col]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Run script for either all of Great Britain; Plymouth only {plymouth}; or Plymouth and 4 similar local authorities {plymouth_similar}; or Plymouth and 5 different local authorities {sampling_areas}. Default to all of GB",
        type=str,
        default="GB",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    import polars as pl

    from asf_heat_pump_suitability.getters import (
        load_geodata,
        load_tree_input,
        load_boundaries,
    )
    from asf_heat_pump_suitability.pipeline.transform import (
        non_residential_entities,
        poi,
    )
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = generate_gdf_uprn_coords(uprns_df)

    # TODO I expect this to be simplified at some point but the if/else block allows us to sample from certain areas for now
    if args.local_authorities.lower() == "plymouth":
        print("Creating residential UPRN dataset for Plymouth Local Authority...")
        grid_squares = config["constant"]["grid_squares"]["plymouth"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las="Plymouth"
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    elif args.local_authorities.lower() == "plymouth_similar":
        print(
            "Creating residential UPRN dataset for Plymouth, Portsmouth, Southampton, Swansea, and Liverpool Local Authorities..."
        )
        grid_squares = config["constant"]["grid_squares"]["plymouth_similar_cities"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=config["constant"]["plymouth_similar_cities"]
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    elif args.local_authorities.lower() == "sampling_areas":
        print(
            "Creating residential UPRN dataset for Bath, Bradford, Glasgow, Manchester, Nottingham, and Plymouth Local Authorities..."
        )
        grid_squares = config["constant"]["grid_squares"]["sampling_areas"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=config["constant"]["sampling_areas"]
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    else:  # All of GB
        # TODO this may not work due to scaling and may require chunking of datasets.
        # TODO Adding here as placeholder to assist scaling later
        print("Creating residential UPRN dataset for all of GB...")
        grid_squares = None

    poi_gdf = load_tree_input.load_gdf_poi()
    poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=poi.load_set_non_domestic_poi_categories(),
    )

    # Get layers required for identifying residential UPRNs
    layers = {
        f"{layer}_gdf": load_tree_input.load_gdf_os_openmap_local_layer(
            layer=layer, grid_squares=grid_squares
        )
        for layer in ["important_building", "railway_station", "building"]
    }

    # Identify assumed non-residential buildings
    non_residential_buildings_gdf = (
        non_residential_entities.generate_gdf_non_residential_buildings(
            **layers, poi_gdf=poi_gdf, uprns_gdf=uprns_gdf
        )
    )

    # Filter UPRNs to assumed residential only
    residential_uprns_gdf = filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=layers["building_gdf"],
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )

    # Save residential UPRNs to S3
    df = pl.from_pandas(
        residential_uprns_gdf[
            [
                "UPRN",
                "X_COORDINATE",
                "Y_COORDINATE",
                "LATITUDE",
                "LONGITUDE",
                "LAD23CD",
                "LAD23NM",
            ]
        ]
    )
    save_utils.save_to_s3(
        df,
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{args.local_authorities}_residential_uprns.parquet",
    )
