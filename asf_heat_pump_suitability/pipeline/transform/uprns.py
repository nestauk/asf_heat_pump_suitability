"""
Functions to transform UPRN data.

Contains script to filter OS UPRNs to residential UPRNs only. UPRNs that meet any of the following criteria are assumed to be
residential:
- UPRNs geolocated inside a building footprint but not in a 'non-residential' building type (see non_residential_entities.py) and not in the non-domestic EPC register
- UPRNs found in the domestic EPC register

To run the script:
python asf_heat_pump_suitability/pipeline/transform/uprns.py --local_authorities LOCAL_AUTHORITIES

Defaults to `GB` (all of Great Britain), but this is not yet implemented.

Set --save to save the outputs to S3. By default, outputs are not saved.
"""

import geopandas as gpd
import pandas as pd
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
    if epc_type == "domestic":
        df = base_getters.load_df_from_s3(
            config["data"]["epc"][epc_type], columns="UPRN"
        )
        before = len(df)

    else:
        # England and Wales EPC data
        df_EW = base_getters.load_df_from_s3(
            config["data"]["epc"][epc_type]["EW"], columns="UPRN"
        )
        # Scotland EPC data
        df_S = (
            base_getters.load_df_from_s3(
                config["data"]["epc"][epc_type]["S"], columns="OSG_REFERENCE_NUMBER"
            )
            .rename({"OSG_REFERENCE_NUMBER": "UPRN"})
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64)
        )

        df = pl.concat([df_EW, df_S])
        before = len(df)

    df = df.with_columns(
        # Remove any invalid UPRNs (i.e. those IDs which are generated in EPC preprocessing generated from concatenating building ref number and address)
        # These are not true UPRNs that can be used in joins across other datasets
        pl.col("UPRN")
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64)
        .alias("UPRN")
    ).drop_nulls()  # TODO: Scotland commercial EPC data has a lot (37 %) of null UPRNs.

    logging.info(
        f"{before - len(df)} invalid UPRNs dropped from {epc_type} EPC register. {len(df)} valid UPRNs remaining"
    )

    return set(df["UPRN"])


def filter_gdf_domestic_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    non_residential_buildings_gdf: gpd.GeoDataFrame,
    local_authority: str | list[str],
    id_col: str = config["constant"]["id"]["building"],
) -> gpd.GeoDataFrame:
    """
    Filter UPRNs to domestic UPRNs only by retaining UPRNs which appear in domestic EPC register, OR are located within
    a building footprint AND are not in the commercial EPC register and / or a building type that is unlikely to contain
    residential properties, e.g. hospital, train station, museum etc, AND (for Plymouth only) are in a building with
    `m2_per_predicted_UPRN` below a defined threshold.

    See analysis in /research/exploratory/domestic_filtering/domestic_building_identification_threshold_selection.py for
    threshold selection for Plymouth.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be filtered.
        buildings_gdf (gpd.GeoDataFrame): all building footprints in area of interest.
        non_residential_buildings_gdf (gpd.GeoDataFrame): polygons of buildings which are unlikely to contain residential
        properties.
        local_authority (str): name of local authority the domestic UPRNs are being identified for.
        id_col (str): name of ID column in `buildings_gdf`. Defaults to ID column defined in config.

    Returns:
        gpd.GeoDataFrame: UPRNs which are assumed to represent domestic properties with their point geometries
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
    domestic_epc_uprns = load_set_valid_epc_uprns(epc_type="domestic")

    domestic_uprn_gdf = uprn_gdf[
        (
            # Filter to UPRNs which are in buildings AND not in non-residential UPRNs list
            (~uprn_gdf["UPRN"].isin(non_residential_uprns))
            & (uprn_gdf["UPRN"].isin(uprns_in_buildings))
        )
        # Or UPRNs which are in domestic EPC register
        | (uprn_gdf["UPRN"].isin(domestic_epc_uprns))
    ]

    # TODO this could be updated to a classification model and scaled
    # This triggers for Plymouth only as the threshold density was calculated from Plymouth data only
    if [la.lower() for la in local_authority_dict["valid_local_authorities"]] == [
        "plymouth"
    ]:
        # Identify large buildings with low UPRN density which will be labelled non-domestic
        non_domestic_buildings_gdf = _generate_gdf_non_domestic_buildings_by_density(
            domestic_uprns_gdf=domestic_uprn_gdf,
            buildings_gdf=buildings_gdf,
            epc_uprns=domestic_epc_uprns,
            id_col=id_col,
        )

        # Remove UPRNs in these buildings from the domestic subset
        non_domestic_uprns = set(
            uprn_gdf.sjoin(
                non_domestic_buildings_gdf, how="inner", predicate="intersects"
            )["UPRN"]
        )

        return domestic_uprn_gdf[~domestic_uprn_gdf["UPRN"].isin(non_domestic_uprns)]

    else:
        return domestic_uprn_gdf


def _generate_gdf_non_domestic_buildings_by_density(
    domestic_uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    epc_uprns: set,
    id_col: str,
    threshold: float = config["constant"]["threshold"]["m2_per_predicted_UPRN"],
) -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame containing footprints of buildings which contain a UPRN predicted to be domestic and have a
    `m2_per_predicted_UPRN` value above the defined threshold. The aim of this function is to remove some of the true
    non-domestic buildings mislabelled as domestic by removing those with large building footprints and low UPRN density,
    unless they contain a UPRN with a domestic EPC record.

    The threshold selected as default (set in config) was determined through analysis that can be found in
    asf_heat_pump_suitability/research/exploratory/domestic_filtering/domestic_building_identification_threshold_selection.py

    Args:
        domestic_uprns_gdf (gpd.GeoDataFrame): UPRNs which are predicted by the pipeline to represent domestic properties with their point geometries.
        buildings_gdf (gpd.GeoDataFrame): all building footprints in area of interest.
        epc_uprns (str): UPRNs with a domestic EPC record.
        id_col (str): name of ID column in `buildings_gdf`. Defaults to ID column defined in config.
        threshold (float): threshold for `m2_per_predicted_UPRN` above which a building is considered to be non-domestic.

    Returns:
        gpd.GeoDataFrame: footprints of buildings containing predicted domestic UPRNs but likely to be non-domestic
    """
    # Join predicted domestic UPRNs to buildings
    _buildings_gdf = buildings_gdf.sjoin(
        domestic_uprns_gdf[["UPRN", "geometry"]], how="inner", predicate="contains"
    ).drop(columns="index_right")

    # Identify EPC UPRNs
    _buildings_gdf["domestic_epc"] = _buildings_gdf["UPRN"].isin(epc_uprns)

    # Get building area
    _buildings_gdf["footprint_area_m2"] = _buildings_gdf.area

    # Get predicted domestic UPRN count per building
    _buildings_gdf = (
        _buildings_gdf.groupby(id_col)
        .agg(
            contains_epc=("domestic_epc", "max"),
            predicted_UPRN_count=("UPRN", "count"),
            footprint_area_m2=("footprint_area_m2", "first"),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )

    # Remove buildings containing a domestic EPC record
    _buildings_gdf = _buildings_gdf[~_buildings_gdf["contains_epc"]]

    # Calculate density measure
    _buildings_gdf["m2_per_predicted_UPRN"] = (
        _buildings_gdf["footprint_area_m2"] / _buildings_gdf["predicted_UPRN_count"]
    )
    # Return buildings with density measure above threshold
    return gpd.GeoDataFrame(
        _buildings_gdf[_buildings_gdf["m2_per_predicted_UPRN"] > threshold].copy(),
        geometry="geometry",
        crs=27700,
    )


def map_dict_uprns_to_building_id(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    id_col: str,
    predicate: str = "intersects",
    max_distance: float = 1,
) -> dict:
    """
    Create a mapping of UPRNs (keys) to the building ID (values) of the building they are located within or intersect with, or the nearest building < 1m away if not located within a building.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs with geospatial point data
        buildings_gdf (gpd.GeoDataFrame): building footprints
        id_col (str): name of building ID column in `buildings_gdf`
        predicate (str): how to join buildings and UPRNs. Can be one of: `intersects`, which joins UPRNs with building footprints
        they intersect with, or `within` which joins UPRNs to building footprints they are located within. Default `intersects`.
        max_distance (float): max distance (metres) from which to join UPRNs to a building footprint. Default 1m.

    Returns:
        dict: mapping of UPRNs to building IDs
    """
    uprns_gdf = geo_utils.verify_gdf_crs(uprns_gdf)
    buildings_gdf = geo_utils.verify_gdf_crs(buildings_gdf)

    # join domestic UPRNs to either the building footprint they are located within or building < max_distance (default 1m) away. They are usually inside a building.

    # first find domestic UPRNs inside buildings to speed up computation time.
    uprns_inside_buildings = uprns_gdf.sjoin(
        buildings_gdf, how="inner", predicate="intersects"
    )

    # find domestic UPRNs not joined to a buildings.
    unmatched_uprns = uprns_gdf[~uprns_gdf["UPRN"].isin(uprns_inside_buildings)]

    # for these UPRNs only, find nearest building footprint < max_distance (m) away
    nearest_buildings_uprns = unmatched_uprns.sjoin_nearest(
        buildings_gdf, how="inner", max_distance=max_distance
    )

    # combine the two gdfs and turn into a dictionary
    uprns_building_dict = (
        (pd.concat([uprns_inside_buildings, nearest_buildings_uprns]))
        .set_index("UPRN")
        .to_dict()[id_col]
    )

    return uprns_building_dict


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        default="GB",
        required=False,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import polars as pl

    from asf_heat_pump_suitability.getters import (
        load_geodata,
        load_boundaries,
    )
    from asf_heat_pump_suitability.pipeline.transform import (
        non_residential_entities,
        poi,
        local_authority,
    )
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    local_authorities = [la.lower() for la in args.local_authorities]

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = generate_gdf_uprn_coords(uprns_df)

    if local_authorities == "gb":  # All of GB
        # TODO this may not work due to scaling and may require chunking of datasets.
        # TODO Adding here as placeholder to assist scaling later
        print("Creating residential UPRN dataset for all of GB...")
        grid_squares = None
    else:  # Specific local authorities (any number of LAs can be specified in config file)
        print(f"Creating residential UPRN dataset for {local_authorities}...")
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=local_authority_dict["valid_local_authorities"]
        )  # TODO: add if statement for running with a test script
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(
            columns="index_right"
        )  # TODO: will have already cut down uprns so no need for this line when running with a test script

    poi_gdf = load_geodata.load_gdf_poi()
    poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=None,
    )
    non_domestic_poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=poi.load_set_non_domestic_poi_categories(),
    )

    # Get layers required for identifying residential UPRNs
    layers = {
        f"{layer}_gdf": load_geodata.load_gdf_os_openmap_layer(
            layer=layer, grid_squares=local_authority_dict["grid_squares"]
        )
        for layer in ["important_building", "railway_station", "building"]
    }

    # Identify assumed non-residential buildings
    non_residential_buildings_gdf = (
        non_residential_entities.generate_gdf_non_residential_buildings(
            **layers,
            non_domestic_poi_gdf=non_domestic_poi_gdf,
            poi_gdf=poi_gdf,
            uprns_gdf=uprns_gdf,
        )
    )

    # Filter UPRNs to assumed domestic only
    domestic_uprns_gdf = filter_gdf_domestic_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=layers["building_gdf"],
        non_residential_buildings_gdf=non_residential_buildings_gdf,
        local_authority=[
            la.lower() for la in local_authority_dict["valid_local_authorities"]
        ],
    )

    # Save residential UPRNs to S3
    df = pl.from_pandas(
        domestic_uprns_gdf[
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

    if args.save:
        save_utils.save_to_s3(
            df,
            config["output"]["dataset"]["domestic_uprns"].format(
                local_authority=local_authority_dict["url_slug"]
            ),
        )
