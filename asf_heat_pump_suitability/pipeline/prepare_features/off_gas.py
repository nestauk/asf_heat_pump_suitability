from typing import List

import polars as pl
import pandas as pd
import geopandas as gpd

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters.get_datasets import get_df_spa_offgasgrid
from asf_heat_pump_suitability.pipeline.transform import uprns


def process_off_gas_data() -> List[str]:
    """
    This function processes the off-gas data by removing spaces from the postcodes and converting them to a list.

    Returns:
        List[str]: The list of processed off-gas postcodes.
    """
    off_gas_df = get_df_spa_offgasgrid()
    off_gas_postcodes = off_gas_df["Post Code"].str.replace(r"\s+", "").to_list()
    return off_gas_postcodes


def add_off_gas_feature(df: pl.DataFrame, off_gas_postcodes: List[str]) -> pl.DataFrame:
    """
    This function adds an 'off_gas' column to a DataFrame based on whether 'POSTCODE' is in off_gas_postcodes.

    Args:
        df (pl.DataFrame): EPC dataset with postcode column
        off_gas_postcodes (List[str]): The list of processed off-gas postcodes.

    Returns:
        pl.DataFrame: EPC dataframe with the added `off_gas` column.
    """
    df = df.with_columns(pl.col("POSTCODE").is_in(off_gas_postcodes).alias("off_gas"))
    return df


def extend_df_off_gas(
    features_df: pl.DataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    code_point_gdf: gpd.GeoDataFrame,
    off_gas_list: list,
    id_col: str = "ID",
    max_distance_m: int = 500,
) -> pl.DataFrame:
    """
    Add boolean column to features_df indicating whether UPRN is in an off-gas postcode.

    `features_df` is expected to have a `POSTCODE` column which is used to label UPRNs with on/off gas where possible.
    For UPRNs without a postcode, the function attempts to assign a postcode based on the nearest UPRN in the same building
    or the nearest code point within a specified distance, and then label off-gas as True/False accordingly.

    Args:
        features_df (pl.DataFrame): dataframe with one row per UPRN and the following columns: UPRN, POSTCODE, and id_col.
        uprns_gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries for each UPRN.
        code_point_gdf (gpd.GeoDataFrame): GeoDataFrame of postcode centroids with geometry column and POSTCODE column.
        off_gas_list (list): list of postcodes that are off-gas.
        id_col (str): column name for building ID. Defaults to "ID".
        max_distance_m (int): maximum distance in metres to assume a UPRN is associated with a postcode if it doesn't have its own postcode. Defaults to 500 metres.

    Returns:
        pl.DataFrame: input features_df with new boolean column `off_gas` indicating whether UPRN is in an off-gas postcode.
    """
    print("Adding off_gas feature to features_df...")
    # Step 0: Use EPC postcode for UPRNs where available
    # Done previously, as features_df already contains the EPC postcode column

    print(
        "Number of UPRNs with POSTCODE using EPC only:",
        features_df.filter(pl.col("POSTCODE").is_not_null()).height,
    )

    # Step 1: Use EPC postcode of nearest UPRN in the same building if available

    # Mapping for non-null POSTCODEs: create mapping between id_col (building ID) and POSTCODE when POSTCODE is not null
    id_postcode_mapping_df = features_df.filter(
        pl.col("POSTCODE").is_not_null()
    ).select(["UPRN", id_col, "POSTCODE"])

    # Add geometry to the mapping
    id_postcode_mapping_gdf = (
        id_postcode_mapping_df.to_pandas()
        .merge(uprns_gdf[["UPRN", "geometry"]], on="UPRN")
        .set_geometry("geometry")
    )

    # This gdf will have UPRN, geometry and id_col (building ID) columns which will be used to find nearest postcode within the same building where available
    features_gdf = (
        uprns_gdf[["UPRN", "geometry"]]
        .merge(features_df[[id_col, "UPRN"]].to_pandas(), on="UPRN", how="right")
        .set_geometry("geometry")
    )

    # Create geodataframe of UPRNs with their mapped closest POSTCODE from the same building, where available
    postcodes_gdf = (
        features_gdf.groupby(id_col, group_keys=False)
        .apply(
            lambda x: gpd.sjoin_nearest(
                x,
                id_postcode_mapping_gdf[id_postcode_mapping_gdf[id_col] == x.name],
                how="left",
                distance_col="distance_to_nearest_postcode_m",
            )
        )
        .sort_values("distance_to_nearest_postcode_m", ascending=True)
        .drop_duplicates(subset=["UPRN_left"])[
            ["UPRN_left", id_col + "_left", "POSTCODE"]
        ]
        .rename(columns={"UPRN_left": "UPRN", id_col + "_left": id_col})
    )

    # Go back to polars
    postcodes_df = pl.from_pandas(postcodes_gdf[["UPRN", id_col, "POSTCODE"]])

    print(
        "Number of UPRNs with POSTCODE after mapping postcodes from same building:",
        postcodes_df.filter(pl.col("POSTCODE").is_not_null()).height,
    )

    # Step 2: Use EPC postcode of nearest code point within a specified distance
    uprns_missing_postcode_df = postcodes_df.filter(
        pl.col("POSTCODE").is_null()
    ).get_column("UPRN")
    uprns_missing_postcode_gdf = uprns_gdf[
        uprns_gdf["UPRN"].isin(uprns_missing_postcode_df)
    ]

    nearest_postcode_df = pl.from_pandas(
        uprns_missing_postcode_gdf.sjoin_nearest(
            code_point_gdf[["POSTCODE", "geometry"]],
            how="left",
            max_distance=max_distance_m,  # maximum distance in metres
            distance_col="distance_to_postcode_m",  # distance in metres
        )
        .sort_values("distance_to_nearest_postcode_m", ascending=True)
        .drop_duplicates(subset=["UPRN"])
        .drop(columns="index_right")[["UPRN", "POSTCODE", "distance_to_postcode_m"]]
    )

    # Step 3: Combine UPRNs with known postcodes and nearest postcodes
    uprn_postcode_map_df = pl.concat(
        [
            postcodes_df.filter(pl.col("POSTCODE").is_not_null()).select(
                ["UPRN", "POSTCODE"]
            ),
            nearest_postcode_df.select(["UPRN", "POSTCODE"]),
        ],
        how="vertical",
    )

    print(
        "Number of UPRNs with POSTCODE after adding nearest code point postcodes:",
        uprn_postcode_map_df.filter(pl.col("POSTCODE").is_not_null()).shape[0],
    )

    # Step 4: Label all UPRNs with on/off gas where possible
    off_gas_df = uprn_postcode_map_df.with_columns(
        # Label postcodes according to on/off gas
        pl.when(pl.col("POSTCODE").is_in(off_gas_list))
        .then(True)
        .otherwise(False)
        .alias("off_gas")
    ).select(["UPRN", "off_gas"])

    # Extend features_df with off_gas flag
    features_df = features_df.join(
        off_gas_df,
        how="left",
        on="UPRN",
    )

    return features_df


def fill_df_missing_postcodes(
    uprns_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    code_point_gdf: gpd.GeoDataFrame,
    max_distance: float = 500,
    id_col: str = config["constant"]["id"]["building"],
) -> pl.DataFrame:

    # Assign building IDs to UPRNs
    building_id = "_internal_building_id"
    uprn_to_building_dict = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf, id_col=id_col
    )
    uprns_gdf[building_id] = uprns_gdf["UPRN"].replace(uprn_to_building_dict)

    # Separate UPRNs into those with and without postcodes
    donors_gdf = uprns_gdf[uprns_gdf["POSTCODE"].notna()].copy()[
        [building_id, "UPRN", "POSTCODE", "geometry"]
    ]
    recipients_gdf = uprns_gdf[uprns_gdf["POSTCODE"].isna()].copy()[
        [building_id, "UPRN", "geometry"]
    ]

    # There will be no recipients if all UPRNs already have a postcode
    if recipients_gdf.empty:
        return donors_gdf

    # RULE 1: Find closest UPRN WITHIN the same building
    rule1_results = []

    # Get recipients that are actually inside a building
    in_building_recipients_gdf = recipients_gdf[recipients_gdf[building_id].notna()]

    # Iterate through every building and assign the closest in-building postcode to UPRNs with no postcode
    for b_id, recipients_in_building_gdf in in_building_recipients_gdf.groupby(
        building_id
    ):
        building_donors_gdf = donors_gdf[donors_gdf[building_id] == b_id]

        if not building_donors_gdf.empty:
            # Find the nearest donor UPRN within the same building
            nearest_postcodes_gdf = recipients_in_building_gdf.sjoin_nearest(
                building_donors_gdf[["POSTCODE", "geometry"]],
                how="left",
            )
            rule1_results.append(nearest_postcodes_gdf)
        else:
            # No donors in this building, they will be processed by Rule 2
            rule1_results.append(recipients_in_building_gdf)

    # Concat rule 1 results for all buildings
    if rule1_results:
        filled_postcodes_gdf = pd.concat(rule1_results)
    else:  # This should only trigger if all the recipient UPRNs are located outside of building footprints
        filled_postcodes_gdf = recipients_gdf.copy()

    # RULE 2: Closest point OUTSIDE/ANYWHERE within max distance
    # Identify new recipient UPRN subset still missing postcodes
    recipients_gdf = filled_postcodes_gdf[
        filled_postcodes_gdf["POSTCODE"].isna()
    ].copy()[["UPRN", "POSTCODE", "geometry"]]
    print(recipients_gdf.head())

    # Triggered unless every UPRN received a postcode from the same building by applying rule 1
    if not recipients_gdf.empty:
        # Combine original UPRNs and code points into one dataframe to identify nearest postcode
        all_donors_gdf = pd.concat(
            [
                donors_gdf[["POSTCODE", "geometry"]],
                code_point_gdf.rename(columns={"postcode": "POSTCODE"})[
                    ["POSTCODE", "geometry"]
                ],
            ]
        )
        print("\n\n")
        print(all_donors_gdf.head())

        # Join postcode from nearest UPRN or code point within specified distance
        nearest_postcodes_gdf = recipients_gdf.sjoin_nearest(
            all_donors_gdf[["POSTCODE", "geometry"]],
            how="left",
            rsuffix="_second_pass",
            max_distance=max_distance,
        )

        print("\n\n")
        print(nearest_postcodes_gdf.head())

        # Fill postcodes missing from the first pass with the second pass
        filled_postcodes_gdf = filled_postcodes_gdf.merge(
            nearest_postcodes_gdf, how="left", on="UPRN"
        )
        filled_postcodes_gdf["POSTCODE"] = filled_postcodes_gdf["POSTCODE"].fillna(
            filled_postcodes_gdf["POSTCODE_second_pass"]
        )

    # Concat original donors and the filled recipients
    uprn_postcodes_gdf = pd.concat(
        [donors_gdf[["UPRN", "POSTCODE"]], filled_postcodes_gdf[["UPRN", "POSTCODE"]]]
    )

    # If there were multiple equidistant UPRNs/code points for a single UPRN, the UPRN will have multiple postcodes
    return pl.from_pandas(uprn_postcodes_gdf)
