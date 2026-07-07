"""
This script codes the decision tree that outputs the most suitable tech for each UPRN given the following inputs:
- whether it is in a city centre or planned HN zone
- maximum contiguous outdoor space size
- whether the property is in a block of flats or not

It then aggregates the most suitable tech information at building footprint level, and assigns a unique solution for each building.

To run the script:

python asf_heat_pump_suitability/pipeline/transform/decision_tree.py --local_authorities LOCAL_AUTHORITIES

Use --save if you want to save the outputs to S3.

Set --release_date to specify the YYYYMMDD dated release directory to read inputs from and
save outputs to. Defaults to today's date.
"""

# package imports
import pandas as pd
import polars as pl
import geopandas as gpd
import argparse

# local imports
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.getters.load_geodata import (
    load_gdf_os_openmap_layer,
)
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.transform import local_authority
from asf_heat_pump_suitability.utils import save_utils

OUTDOOR_SPACE_THRESHOLD_M2 = config["constant"]["threshold"][
    "outdoor_space_threshold_m2"
]
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
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    return parser.parse_args()


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
                "assigned_tech": TECH_TYPES["heat_networks"],
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
                    "assigned_tech": f"{TECH_TYPES['individual']} or {TECH_TYPES['heat_networks']}",
                    "decision_tree_path": "Not in block of flats. Unknown outdoor space in city centre",
                }
            elif outdoor_space > OUTDOOR_SPACE_THRESHOLD_M2.get("within_hn_zone"):
                return {
                    "assigned_tech": TECH_TYPES["individual"],
                    "decision_tree_path": "3. not in blocks of flats, in city centre, large outdoor space",
                }
            else:
                return {
                    "assigned_tech": TECH_TYPES["heat_networks"],
                    "decision_tree_path": "4. not in blocks of flats, in city centre, small or no outdoor space",
                }
        else:
            if not outdoor_space:
                return {
                    "assigned_tech": f"{TECH_TYPES['individual']} or {TECH_TYPES['networked']}",
                    "decision_tree_path": "Not in block of flats. Unknown outdoor space not in city centre/ HN zone",
                }
            elif outdoor_space > OUTDOOR_SPACE_THRESHOLD_M2.get("outside_hn_zone"):
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
    tech_gdf: gpd.GeoDataFrame, id_col: str
) -> pd.DataFrame:
    """
    Assigns a single most suitable tech solution to each building footprint.

    - Identifies the different solutions for each building (the set of solutions for UPRNs within the same building footprint)
    - Where only 1 solution exists for a building footprint, assign that solution as the most suitable tech for the building footprint
    - Where multiple solutions exist for a building footprint, assign a unique solution based on the combination of solutions in the set and the median outdoor space of properties within the building footprint.
    This function aggregates the most suitable tech from individual UPRNs to the building level.

    Multiple solutions within the same building might occur when a building contains multiple UPRNs and the decision tree assigns different technologies to at least 2 of them.
    A few examples of where this happens:
    - a building that sits at the edge of a HN zones, might have properties in the building assigned district heat network and others networked GSHP.
    - a building footprint consisting of a row of terraced houses, where some properties have large outdoor space and others small outdoor space.

    Args:
    tech_gdf (gpd.GeoDataFrame): GeoDataFrame containing UPRN-level most suitable tech. Must contain the following columns:
        * `UPRN`: Unique identifier for the property.
        * `assigned_tech`: The most suitable tech assigned by the decision tree.
        * `id_col`: Unique identifier for the building footprint.
        * `max_contiguous_outdoor_space_area_m2`: Maximum contiguous outdoor space in metres squared.
    id_col (str): The name of the column in `tech_gdf` that contains the unique identifier for the building footprint (e.g. "ID").

    Returns:
        pd.DataFrame: Buildings with a single assigned technology.

    Raises:
        ValueError: If `tech_gdf` is missing any of the required columns.
    """

    for col in [
        "UPRN",
        "assigned_tech",
        id_col,  # the building ID
        "max_contiguous_outdoor_space_area_m2",
    ]:
        if col not in tech_gdf.columns:
            raise ValueError(
                f"Input GeoDataFrame must contain the following columns: ['UPRN', 'assigned_tech', '{id_col}', 'max_contiguous_outdoor_space_area_m2', 'in_hn_zone']."
            )

    tech_df = pl.from_pandas(
        pd.DataFrame(
            tech_gdf[
                [
                    "UPRN",
                    "assigned_tech",
                    id_col,  # the building ID
                    "max_contiguous_outdoor_space_area_m2",
                ]
            ]
        )
        # This will drop valid EPC UPRNs with no building footprint.
        # These need to be mapped to their nearest building footprints to be included.
        # This is required to prevent the creation of a row with no geometry
    ).drop_nulls(subset=id_col)

    # Create df with set of most suitable tech per building footprint
    solutions_per_footprint_df = (
        # Aggregate at building footprint level to identify the set of most suitable tech for properties within the same building
        tech_df.group_by(
            id_col
        )  # grouping by building ID to aggregate to building footprint level
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
        [id_col, "assigned_tech"]
    )

    # For building footprints with only 1 solution, assign that solution as the unique solution for the building
    buildings_with_single_solution_df = (
        solutions_per_footprint_df.filter(pl.col("n_solutions") == 1)
        .with_columns(assigned_tech=pl.col("assigned_tech").list.get(0))
        .select([id_col, "assigned_tech"])
    )

    # Combine the dataframes of building footprints with multiple solutions and single solution to get the final dataframe with a unique assigned solution for each building footprint
    solutions_per_footprint_df = pl.concat(
        [buildings_with_multiple_solutions_df, buildings_with_single_solution_df]
    )

    # Convert back to Pandas df to be merged to GeoDataFrames in the next steps of the pipeline
    solutions_per_footprint_df = solutions_per_footprint_df.to_pandas()

    return solutions_per_footprint_df


def assign_df_unique_solution(solutions_per_footprint_df: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a unique solution for each building footprint based on the combination of solutions in the set
    and the median outdoor space of properties within the building footprint, following the logic below:

    - If the set of solutions contains both "District heat network" and "Networked heat pump", assign "Communal".
    - Else if the set of solutions contains "District heat network", assign "District heat network".
    - Else if the set of solutions contains "Networked heat pump", assign "Networked heat pump".
    - Else if the set of solutions contains "Communal", assign "Communal"
    - Else if the set of solutions contains both "Individual heat pump" and "Networked heat pump", assign:
        - "Individual heat pump" if at least 50% of properties in the building footprint have outdoor space data available and the median outdoor space area is greater than threshold defined for properties outside HN zones/ city centres
        - "Networked heat pump" otherwise
    - Else if the set of solutions contains both "Individual heat pump" and "District heat network", assign:
        - "Individual heat pump" if at least 50% of properties in the building footprint have outdoor space data available and the median outdoor space area is greater than threshold defined for properties within HN zones/ city centres
        - "District heat network" otherwise
    - Else, assign "Unexpected combination of solutions in building footprint"

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
                pl.col("assigned_tech").list.contains(TECH_TYPES["heat_networks"])
                & pl.col("assigned_tech").list.contains(TECH_TYPES["networked"])
            )
            .then(pl.lit(TECH_TYPES["communal"]))
            .when(pl.col("assigned_tech").list.contains(TECH_TYPES["heat_networks"]))
            .then(pl.lit(TECH_TYPES["heat_networks"]))
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
                        pl.col("median_contiguous_outdoor_space_area_m2")
                        > OUTDOOR_SPACE_THRESHOLD_M2.get("outside_hn_zone")
                    )  # Using your column name from previous step
                )
                .then(pl.lit(TECH_TYPES["individual"]))
                .otherwise(pl.lit(TECH_TYPES["networked"]))
            )
            .when(
                pl.col("assigned_tech").list.contains(
                    f"{TECH_TYPES['individual']} or {TECH_TYPES['heat_networks']}"
                )
            )
            .then(
                pl.when(
                    (pl.col("perc_properties_available_outdoor_space_data") >= 50)
                    & (
                        pl.col("median_contiguous_outdoor_space_area_m2")
                        > OUTDOOR_SPACE_THRESHOLD_M2.get("within_hn_zone")
                    )
                )
                .then(pl.lit(TECH_TYPES["individual"]))
                .otherwise(pl.lit(TECH_TYPES["heat_networks"]))
            )
            .otherwise(pl.lit("Unexpected combination"))
        )
    )

    return solutions_per_footprint_df


def identify_gdf_tuple_most_suitable_tech_uprn_and_building(
    local_authorities: str,
    buildings_gdf: gpd.GeoDataFrame,
    id_col: str,
    uprns_gdf: gpd.GeoDataFrame,
    save: bool,
    release_date: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Main function to identify the most suitable tech for each UPRN and building in the specified local authority or authorities.

    Saves outputs to S3 if specified, to the dated release directory for `release_date`.

    Args:
        local_authorities (str): local authority slug to save decision tree outputs to.
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame with building footprints.
        id_col (str): The name of the column in `buildings_gdf` that contains the unique identifier for the building footprint (e.g. "ID").
        uprns_gdf (gpd.GeoDataFrame): GeoDataFrame with UPRN data.
        save (bool): Whether to save outputs to S3.
        release_date (str): release date in YYYYMMDD format used for the dated output
            directory, as returned by `save_utils.get_str_release_date`.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: A tuple containing:
            - GeoDataFrame with UPRN-level most suitable tech and decision tree path.
            - GeoDataFrame with building footprint-level most suitable tech.
    """
    # Join corresponding building geometry to each UPRN
    buildings_gdf["building_geometry"] = buildings_gdf["geometry"]
    uprns_gdf = uprns_gdf.merge(
        buildings_gdf[[id_col, "building_geometry"]],
        how="left",
        on=id_col,
    )

    print("Number of building IDs: ", uprns_gdf[id_col].nunique())
    print("Number of building geometries: ", uprns_gdf["building_geometry"].nunique())

    print(
        "Number of UPRNs without building geometry: ",
        len(uprns_gdf[uprns_gdf["building_geometry"].isnull()]),
    )
    print(
        "Number of UPRNs without building ID: ",
        len(uprns_gdf[uprns_gdf[id_col].isnull()]),
    )
    print(
        "Number of UPRNs with ID but no building geometry: ",
        len(
            uprns_gdf[
                (uprns_gdf["building_geometry"].isnull())
                & (uprns_gdf[id_col].notnull())
            ]
        ),
    )

    # Identify most suitable tech for each UPRN
    decision_tree_outputs = uprns_gdf.apply(
        lambda x: identify_dict_most_suitable_tech(
            x["in_block_of_flats"],
            x["max_contiguous_outdoor_space_area_m2"],
            x["in_city_centre_or_hn_zone"],
        ),
        axis=1,
        result_type="expand",
    )

    # Create new columns for `assigned_tech` and `decision_tree_path` based on the most suitable solutions dictionary
    uprns_gdf = pd.concat([uprns_gdf, decision_tree_outputs], axis=1)

    # Identify set of most suitable tech per building
    solutions_per_footprint_df = identify_df_building_most_suitable_tech(
        uprns_gdf, id_col
    )

    # Re-assign most suitable tech to each UPRN based on decision for each building
    uprns_gdf = uprns_gdf.drop(columns=["assigned_tech"]).merge(
        solutions_per_footprint_df[[id_col, "assigned_tech"]],
        on=id_col,
        how="left",
    )

    solutions_per_footprint_gdf = solutions_per_footprint_df.merge(
        buildings_gdf[[id_col, "geometry"]],
        on=id_col,
        how="left",
    )

    solutions_per_footprint_gdf = gpd.GeoDataFrame(
        solutions_per_footprint_gdf, geometry="geometry", crs="EPSG:27700"
    )

    print("Number of building IDS ", solutions_per_footprint_gdf[id_col].nunique())
    print(
        "Number of building geometries: ",
        solutions_per_footprint_gdf.geometry.nunique(),
    )

    if save:
        uprns_gdf.to_parquet(
            save_utils.get_str_output_path(
                "uprns_most_suitable_tech",
                release_date=release_date,
                local_authorities=local_authorities,
            )
        )
        solutions_per_footprint_gdf.to_parquet(
            save_utils.get_str_output_path(
                "buildings_most_suitable_tech",
                release_date=release_date,
                local_authorities=local_authorities,
            )
        )

    return uprns_gdf, solutions_per_footprint_gdf


if __name__ == "__main__":
    args = parse_arguments()
    local_authorities = args.local_authorities

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    # TODO: create getters for footprints & uprns in future
    buildings_gdf = load_gdf_os_openmap_layer(
        layer="building",
        grid_squares=local_authority_dict["grid_squares"],
    )

    release_date = save_utils.get_str_release_date(args.release_date)

    uprns_with_features_df = pl.read_parquet(
        save_utils.get_str_output_path(
            "domestic_uprns_with_features",
            release_date=release_date,
            check_exists=True,
            local_authority=local_authority_dict["url_slug"],
        )
    )
    uprns_with_features_gdf = uprns.generate_gdf_uprn_coords(df=uprns_with_features_df)

    # TODO: move this to add_features.py
    uprns_with_features_gdf["in_city_centre_or_hn_zone"] = (
        uprns_with_features_gdf["in_city_centre"]
        | uprns_with_features_gdf["in_hn_zone"]
    )

    identify_gdf_tuple_most_suitable_tech_uprn_and_building(
        local_authorities=local_authority_dict["url_slug"],
        buildings_gdf=buildings_gdf,
        id_col="ID",
        uprns_gdf=uprns_with_features_gdf,
        save=args.save,
        release_date=release_date,
    )
