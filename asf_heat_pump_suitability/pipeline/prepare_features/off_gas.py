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
        .sort_values("distance_to_postcode_m", ascending=True)
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
    """
    Fill missing postcodes for UPRNs. In the first instance, a missing postcode is filled with that of the closest UPRN within
    the same building if available. If unavailable, a missing postcode is filled with that of the closest UPRN OR
    postcode point within the `max_distance`. In cases where a UPRN is joined to multiple postcodes, the postcode with
    the maximum number of matches is selected. If there are multiple maximum matches, then the first postcode alphabetically
    in the group is selected.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs with point geometries and "POSTCODE" column.
        buildings_gdf (gpd.GeoDataFrame): building footprints in area of interest.
        code_point_gdf (gpd.GeoDataFrame): postcode point geometries.
        max_distance (float): maximum search distance for nearest postcode in metres. Default 500m.
        id_col (str): name of building ID column in `buildings_gdf`.

    Returns:
        pl.DataFrame: UPRNs with filled postcode
    """
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

    filled_postcodes_gdf = _fillna_gdf_closest_pcd_in_building(
        donors_gdf=donors_gdf, recipients_gdf=recipients_gdf, building_id=building_id
    )

    # RULE 1: Find closest UPRN WITHIN the same building
    # Identify new recipient UPRN subset still missing postcodes
    recipients_gdf = filled_postcodes_gdf[
        filled_postcodes_gdf["POSTCODE"].isna()
    ].copy()[["UPRN", "geometry"]]

    # RULE 2: Closest point OUTSIDE/ANYWHERE within max distance
    # Triggered unless every UPRN received a postcode from the same building by applying rule 1
    if not recipients_gdf.empty:
        nearest_postcodes_gdf = _fillna_gdf_closest_pcd(
            donors_gdf=donors_gdf,
            recipients_gdf=recipients_gdf,
            code_point_gdf=code_point_gdf,
            max_distance=max_distance,
        )

        # Concat postcodes filled from first and second pass
        filled_postcodes_gdf = pd.concat([filled_postcodes_gdf, nearest_postcodes_gdf])

    # If there were multiple equidistant UPRNs/code points for a single UPRN, the UPRN will have multiple postcodes
    # We don't know which is the true postcode in these cases so we will drop the duplicates reproducibly
    filled_postcodes_df = _deduplicate_df_postcodes_per_uprn(
        filled_postcodes_gdf[["UPRN", "POSTCODE"]]
    )

    # Concat original donors and the filled recipients
    return pl.from_pandas(
        pd.concat(
            [
                donors_gdf[["UPRN", "POSTCODE"]],
                filled_postcodes_df[["UPRN", "POSTCODE"]],
            ]
        )
    )


def _fillna_gdf_closest_pcd_in_building(
    donors_gdf: gpd.GeoDataFrame, recipients_gdf: gpd.GeoDataFrame, building_id: str
) -> gpd.GeoDataFrame:
    """
    Fill missing postcodes with the closest known postcode in the same building.

    Args:
        donors_gdf (gpd.GeoDataFrame): UPRNs with known postcodes. Must contain point geometries and "POSTCODE", and building_id columns.
        recipients_gdf (gpd.GeoDataFrame): UPRNs with missing postcodes. Must contain point geometries and building_id columns.
        building_id (str): name of building ID column.

    Returns:
        gpd.GeoDataFrame: recipient UPRNs with filled postcode data where available.
    """
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
        return pd.concat(rule1_results)
    else:  # This should only trigger if all the recipient UPRNs are located outside of building footprints
        return recipients_gdf.copy()


def _fillna_gdf_closest_pcd(
    donors_gdf: gpd.GeoDataFrame,
    recipients_gdf: gpd.GeoDataFrame,
    code_point_gdf: gpd.GeoDataFrame,
    max_distance: float,
) -> gpd.GeoDataFrame:
    """
    Fill missing postcodes with the closest known postcode.

    Args:
        donors_gdf (gpd.GeoDataFrame): UPRNs with known postcodes. Must contain point geometries and "POSTCODE" column.
        recipients_gdf (gpd.GeoDataFrame): UPRNs with missing postcodes. Must contain point geometries.
        code_point_gdf (gpd.GeoDataFrame): postcode point geometries.
        max_distance (float): maximum search distance for nearest postcode.

    Returns:
        gpd.GeoDataFrame: recipient UPRNs with filled postcode data where available.
    """
    # Combine original UPRNs and code points into one dataframe to identify nearest postcode
    all_donors_gdf = pd.concat(
        [
            donors_gdf[["POSTCODE", "geometry"]],
            code_point_gdf.rename(columns={"postcode": "POSTCODE"})[
                ["POSTCODE", "geometry"]
            ],
        ]
    )

    # Join postcode from nearest UPRN or code point within specified distance
    return recipients_gdf.sjoin_nearest(
        all_donors_gdf[["POSTCODE", "geometry"]],
        how="left",
        max_distance=max_distance,
    )


def _deduplicate_df_postcodes_per_uprn(uprn_postcode_df: pd.DataFrame) -> pl.DataFrame:
    """
    Deduplicate postcodes per UPRN. The postcode with the maximum number of matches to the UPRN is selected. If there
    are multiple maximum matches, then the first postcode alphabetically in the group is selected.

    Args:
        uprn_postcode_df (pd.DataFrame): UPRNs with matching nearest postcodes. Must contain 'UPRN' and 'POSTCODE' columns.

    Returns:
        pl.DataFrame: one row per UPRN with nearest postcode
    """
    # Get the count of each postcode per UPRN
    uprn_postcode_df = (
        uprn_postcode_df.groupby(["UPRN", "POSTCODE"])
        .size()
        .reset_index(name="pcd_count")
    )

    # Add a column with the max postcode count per UPRN
    uprn_postcode_df["pcd_count_max"] = uprn_postcode_df.groupby(["UPRN"])[
        "pcd_count"
    ].transform("max")
    uprn_postcode_df = uprn_postcode_df[
        uprn_postcode_df["pcd_count"] == uprn_postcode_df["pcd_count_max"]
    ]

    # Sort to deduplicate reproducibly
    return uprn_postcode_df.sort_values(by="POSTCODE").drop_duplicates(subset="UPRN")
