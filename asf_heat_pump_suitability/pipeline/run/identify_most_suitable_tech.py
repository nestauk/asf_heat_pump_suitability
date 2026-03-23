"""
This script codes the decision tree that outputs the most suitable tech for each UPRN given the following inputs:
- whether it is in a city centre or planned HN zone
- maximum contiguous outdoor space size
- whether the property is in a block of flats or not

It then aggregates the most suitable tech information at building footprint level, and assigns a unique solution for each building.

To run the script:

python asf_heat_pump_suitability/pipeline/run/identify_most_suitable_tech.py --local_authorities LOCAL_AUTHORITIES

LOCAL AUTHORITIES can be one of the following, as defined in the `constant` section of base.yaml:
- plymouth
- plymouth_similar_cities
- sampling_areas
- greater_manchester_las

Use --save if you want to save the outputs to S3.
"""

# package imports
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import argparse
from shapely import wkb

# local imports
from asf_heat_pump_suitability.pipeline.transform.uprns import generate_gdf_uprn_coords
from asf_heat_pump_suitability.getters.load_tree_input import (
    load_gdf_os_openmap_local_layer,
)
from asf_heat_pump_suitability import config

TECH_TYPES = config["constant"]["tech_types"]


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


def load_df_uprn_data(local_authorities: str) -> gpd.GeoDataFrame:
    """
    Loads UPRN level data with relevant features for the decision tree, including:
    - whether the UPRN is in a block of flats
    - maximum contiguous outdoor space area in metres squared
    - whether the UPRN is in a city centre or planned HN zone

    Creates new column `in_city_centre_or_hn_zone` which indicates whether the UPRN is in the city centre or in a planned HN zone.

    Args:
        local_authorities (str): Local authority or authorities.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with UPRNs and respective features.
    """

    uprns_with_features = pl.read_parquet(
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_with_features.parquet"
    )
    uprns_with_features = generate_gdf_uprn_coords(df=uprns_with_features)

    uprns_with_features["in_city_centre_or_hn_zone"] = (
        uprns_with_features["in_city_centre"] | uprns_with_features["in_hn_zone"]
    )

    return uprns_with_features


def extend_gdf_building_footprints(
    gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Extends the GeoDataFrame with building footprint polygons

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with UPRN data.
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame with building footprints.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with building footprint polygons added.
    """
    # Creates a copy of the (building) geometry column to keep after the spatial join
    gdf["uprn_geometry"] = gdf.geometry
    buildings_gdf["building_geometry"] = buildings_gdf["geometry"]

    # Spatial join to add building footprints geometry to the UPRN geodataframe
    # based on whether the UPRN point is within the building footprint polygon
    gdf = gdf.sjoin(
        buildings_gdf[["geometry", "building_geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    return gdf


def extend_gdf_hn_zone_geometry(
    gdf: gpd.GeoDataFrame, hn_zones_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Extends the GeoDataFrame with heat network zone information.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with UPRN data.
        hn_zones_gdf (gpd.GeoDataFrame): GeoDataFrame with heat network zones.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with heat network zone information.
    """
    # Adds a column to indicate if the property is within a DESNZ heat network zone
    hn_zones_gdf["desnz_hn_zone"] = True

    # Creates a copy of the geometry column to keep after the spatial join
    hn_zones_gdf["desnz_hn_zone_geometry"] = hn_zones_gdf["geometry"]

    # Spatial join to add heat network zone geometry to the UPRN geodataframe
    # based on whether the UPRN point is within the heat network zone polygon
    gdf = gdf.sjoin(
        hn_zones_gdf[["geometry", "desnz_hn_zone_geometry"]],
        how="left",
        predicate="within",
    )

    return gdf


def identify_dict_most_suitable_tech(
    in_block_of_flats: bool, outdoor_space: float, city_centre_or_hnz: bool
) -> dict:
    """
    Defines the decision tree to identify:
    - most suitable low carbon heating solutions for each UPRN.
    - the path taken in the decision tree.

    Args:
        in_block_of_flats (bool): Whether UPRN is in a block of flats.
        outdoor_space (float): Maximum contiguous outdoor space in metres squared.
        city_centre_or_hnz (bool): Whether the UPRN is in the city centre or in a planned heat network zone.

    Returns:
        dict: A dictionary with the most suitable heating solution and the path taken in the decision tree.
    """

    if in_block_of_flats:
        if city_centre_or_hnz:
            return {
                "assigned_tech": TECH_TYPES["heat_network"],
                "decision_tree_path": "1. blocks of flats, in HNZ/ city centre",
            }
        else:
            return {
                "assigned_tech": TECH_TYPES["communal"],
                "decision_tree_path": "2. blocks of flats, not in  HNZ/ city centre",
            }
    else:
        if city_centre_or_hnz:
            if not outdoor_space:
                return {
                    "assigned_tech": f"{TECH_TYPES['individual']} or {TECH_TYPES['heat_network']}",
                    "decision_tree_path": "Not in block of flats. Unknown outdoor space in city centre",
                }
            elif outdoor_space > 70:
                return {
                    "assigned_tech": TECH_TYPES["individual"],
                    "decision_tree_path": "3. not in blocks of flats, in city centre, large outdoor space",
                }
            else:
                return {
                    "assigned_tech": TECH_TYPES["heat_network"],
                    "decision_tree_path": "4. not in blocks of flats, in city centre, small or no outdoor space",
                }
        else:
            if not outdoor_space:
                return {
                    "assigned_tech": f"{TECH_TYPES['individual']} or {TECH_TYPES['networked']}",
                    "decision_tree_path": "Not in block of flats. Unknown outdoor space not in city centre/ HN zone",
                }
            elif outdoor_space > 30:
                return {
                    "assigned_tech": TECH_TYPES["individual"],
                    "decision_tree_path": "5. not in blocks of flats, not in city centre/ HN zone, large outdoor space",
                }
            else:
                return {
                    "assigned_tech": TECH_TYPES["networked"],
                    "decision_tree_path": "6. not in blocks of flats, not in city centre/ HN zone, small/no outdoor space",
                }


def identify_df_building_most_suitable_tech(
    uprns_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Identifies sets of most suitable tech per building footprint, and assigns a unique solution for each building
    based on the combination of solutions in the set (for the UPRNs in the building).

    Args:
        uprns_gdf (gpd.GeoDataFrame): GeoDataFrame with UPRN data.

    Returns:
       gpd.GeoDataFrame: building footprints with assigned most suitable tech.
    """

    # Convert building geometry to WKB format before converting to Polars DataFrame
    uprns_gdf["building_geometry"] = uprns_gdf["building_geometry"].to_wkb()

    uprns_df = pl.from_pandas(
        pd.DataFrame(
            uprns_gdf[
                [
                    "UPRN",
                    "assigned_tech",
                    "building_geometry",
                    "max_contiguous_outdoor_space_area_m2",
                    "in_hn_zone",
                ]
            ]
        )
    )

    # Create df with set of most suitable tech per building footprint
    solutions_per_footprint_df = (
        # Aggregate at building footprint level to identify the set of most suitable tech for properties within the same building
        uprns_df.group_by("building_geometry")
        # Calculate the median outdoor space and percentage of properties with outdoor space data for each building footprint
        # to inform the decision on assigning a unique solution for each building
        .agg(
            assigned_tech=pl.col("assigned_tech").unique(),
            median_contiguous_outdoor_space_area_m2=pl.col(
                "max_contiguous_outdoor_space_area_m2"
            ).median(),
            n_properties_available_outdoor_space_data=pl.col(
                "max_contiguous_outdoor_space_area_m2"
            )
            .drop_nulls()
            .count(),
            n_properties=pl.col("UPRN").count(),
            building_in_hn_zone=pl.col("in_hn_zone").any(),
            n_solutions=pl.col("assigned_tech").n_unique(),
        )
    )

    # Dealing with building footprints with more than 1 solution in the set of most suitable tech for the building footprint
    buildings_with_multiple_solutions_df = (
        solutions_per_footprint_df
        # Filter for footprints with more than 1 solution
        .filter(pl.col("n_solutions") > 1)
        # Calculate the percentage of properties with available outdoor space data for each building footprint
        .with_columns(
            perc_properties_available_outdoor_space_data=(
                pl.col("n_properties_available_outdoor_space_data")
                / pl.col("n_properties")
                * 100
            )
        ).drop(["n_solutions"])
    )

    buildings_with_multiple_solutions_df = assign_df_unique_solution(
        buildings_with_multiple_solutions_df
    )
    buildings_with_multiple_solutions_df = buildings_with_multiple_solutions_df.select(
        ["building_geometry", "assigned_tech", "building_in_hn_zone"]
    )

    # For building footprints with only 1 solution, assign that solution as the unique solution for the building
    buildings_with_single_solution_df = (
        solutions_per_footprint_df.filter(pl.col("n_solutions") == 1)
        .with_columns(assigned_tech=pl.col("assigned_tech").list.get(0))
        .select(["building_geometry", "assigned_tech", "building_in_hn_zone"])
    )

    # Combine the dataframes of building footprints with multiple solutions and single solution to get the final dataframe with a unique assigned solution for each building footprint
    solutions_per_footprint_df = pl.concat(
        [buildings_with_multiple_solutions_df, buildings_with_single_solution_df]
    )

    # Convert back to GeoDataFrame
    solutions_per_footprint_df = solutions_per_footprint_df.to_pandas()
    solutions_per_footprint_df["building_geometry"] = solutions_per_footprint_df[
        "building_geometry"
    ].apply(wkb.loads)
    solutions_per_footprint_gdf = gpd.GeoDataFrame(
        solutions_per_footprint_df, geometry="building_geometry", crs="EPSG:27700"
    )

    return solutions_per_footprint_gdf


def assign_df_unique_solution(solutions_per_footprint_df: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a unique solution for each building footprint based on the combination of solutions in the set
    and the median outdoor space of properties within the building footprint.

    Args:
        solutions_per_footprint_df (pl.DataFrame): DataFrame with the set of most suitable tech for each building footprint, median outdoor space, and percentage of properties with outdoor space data.

    Returns:
        pl.DataFrame: DataFrame with assigned unique solution for each building footprint.
    """
    # Assign a unique solution for each building footprint based on the combination of solutions in the set
    solutions_per_footprint_df = solutions_per_footprint_df.with_columns(
        assigned_tech=(
            # this happens at the edge of HN zones
            # where some properties are assigned district heat network and some networked GSHP
            # due to being just outside the HN zone boundary
            pl.when(
                pl.col("assigned_tech").list.contains(TECH_TYPES["heat_network"])
                & pl.col("assigned_tech").list.contains(TECH_TYPES["networked"])
            )
            .then(pl.lit(TECH_TYPES["communal"]))
            .when(pl.col("assigned_tech").list.contains(TECH_TYPES["heat_network"]))
            .then(pl.lit(TECH_TYPES["heat_network"]))
            .when(pl.col("assigned_tech").list.contains(TECH_TYPES["networked"]))
            .then(pl.lit(TECH_TYPES["networked"]))
            .when(pl.col("assigned_tech").list.contains(TECH_TYPES["communal"]))
            .then(pl.lit(TECH_TYPES["communal"]))
            .when(
                pl.col("assigned_tech").list.contains(
                    f"{TECH_TYPES['individual']} or {TECH_TYPES['networked']}"
                )
            )
            .then(
                pl.when(
                    (pl.col("perc_properties_available_outdoor_space_data") >= 50)
                    & (
                        pl.col("median_contiguous_outdoor_space_area_m2") > 30
                    )  # Using your column name from previous step
                )
                .then(pl.lit(TECH_TYPES["individual"]))
                .otherwise(pl.lit(TECH_TYPES["networked"]))
            )
            .when(
                pl.col("assigned_tech").list.contains(
                    f"{TECH_TYPES['individual']} or {TECH_TYPES['heat_network']}"
                )
            )
            .then(
                pl.when(
                    (pl.col("perc_properties_available_outdoor_space_data") >= 50)
                    & (pl.col("median_contiguous_outdoor_space_area_m2") > 70)
                )
                .then(pl.lit(TECH_TYPES["individual"]))
                .otherwise(pl.lit(TECH_TYPES["heat_network"]))
            )
            .otherwise(pl.lit("Unexpected combination"))
        )
    )

    return solutions_per_footprint_df


def identify_most_suitable_tech_uprn_and_building(
    local_authorities: str, save: bool
) -> None:
    """
    Main function to identify the most suitable tech for each UPRN and building in the specified local authority or authorities.

    Saves outputs to S3 if specified.

    Args:
        local_authorities (str): Local authority or authorities.
        save (bool): Whether to save outputs to S3.
    """

    building_footprints = load_gdf_os_openmap_local_layer(
        layer="building",
        grid_squares=config["constant"][local_authorities]["grid_squares"],
    )

    uprns_gdf = load_df_uprn_data(local_authorities)
    uprns_gdf = extend_gdf_building_footprints(uprns_gdf, building_footprints)

    # Identify most suitable tech for each UPRN
    uprns_gdf["most_suitable_solutions"] = uprns_gdf.apply(
        lambda x: identify_dict_most_suitable_tech(
            x["in_block_of_flats"],
            x["max_contiguous_outdoor_space_area_m2"],
            x["in_city_centre_or_hn_zone"],
        ),
        axis=1,
    )

    # Create new columns for assigned tech and decision tree path based on the most suitable solutions dictionary
    uprns_gdf["assigned_tech"] = [
        x["assigned_tech"] for x in uprns_gdf["most_suitable_solutions"]
    ]
    uprns_gdf["decision_tree_path"] = [
        x["decision_tree_path"] for x in uprns_gdf["most_suitable_solutions"]
    ]

    # Identify set of most suitable tech per building
    solutions_per_footprint_gdf = identify_df_building_most_suitable_tech(uprns_gdf)

    # Re-assign most suitable tech to each UPRN based on decision for each building
    uprns_gdf = uprns_gdf.drop(columns=["assigned_tech"]).merge(
        solutions_per_footprint_gdf[["building_geometry", "assigned_tech"]],
        on="building_geometry",
        how="left",
    )

    # Create new columns in uprns_gdf and solutions_per_footprint_gdf
    # If the building is in a HN zone, assign "District heat network" as the most suitable tech, regardless of the assigned tech based on the decision tree
    uprns_gdf["assigned_tech_hn_precedence"] = np.where(
        uprns_gdf["in_hn_zone"], TECH_TYPES["heat_network"], uprns_gdf["assigned_tech"]
    )

    solutions_per_footprint_gdf["assigned_tech_hn_precedence"] = np.where(
        solutions_per_footprint_gdf["building_in_hn_zone"],
        TECH_TYPES["heat_network"],
        solutions_per_footprint_gdf["assigned_tech"],
    )

    # Drop unecessary columns before saving outputs to S3
    uprns_gdf = uprns_gdf.drop(columns=["in_hn_zone"])
    solutions_per_footprint_gdf = solutions_per_footprint_gdf.drop(
        columns=["building_in_hn_zone"]
    )

    if args.save:
        uprns_gdf.to_parquet(
            f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_uprns_most_suitable_tech.parquet"
        )
        solutions_per_footprint_gdf.to_parquet(
            f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_building_most_suitable_tech.parquet"
        )


if __name__ == "__main__":
    args = parse_arguments()
    local_authorities = args.local_authorities

    identify_most_suitable_tech_uprn_and_building(local_authorities, args.save)
