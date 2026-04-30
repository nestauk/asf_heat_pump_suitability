from typing import List
from asf_heat_pump_suitability.getters.get_datasets import get_df_spa_offgasgrid
import polars as pl
import geopandas as gpd


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
    code_point_df: gpd.GeoDataFrame,
    off_gas_list: list,
    max_distance_m: int = 500,
) -> pl.DataFrame:
    """
    Add boolean column to features_df indicating whether UPRN is in an off-gas postcode.

    Args:
        features_df (pl.DataFrame): dataframe with one row per UPRN, UPRN column and POSTCODE column.
        uprns_gdf (gpd.GeoDataFrame): GeoDataFrame with point geometries for each UPRN.
        code_point_df (gpd.GeoDataFrame): GeoDataFrame of postcode centroids with geometry column and POSTCODE column.
        off_gas_list (list): list of postcodes that are off-gas.
        max_distance_m (int): maximum distance in metres to assume a UPRN is associated with a postcode if it doesn't have its own postcode. Defaults to 500 metres.

    Returns:
        pl.DataFrame: input features_df with new boolean column `off_gas` indicating whether UPRN is in an off-gas postcode.
    """
    # Step 0: Use EPC postcode for UPRNs where available
    # Done previously, as we features_df already contains the EPC postcode column

    # Step 1: Use EPC postcode of nearest UPRN in the same building if available

    # create mapping between ID and POSTCODE when POSTCODE is not null
    id_postcode_mapping_df = (
        features_df.filter(pl.col("POSTCODE").is_not_null())
        .select(["ID", "POSTCODE"])
        .rename({"POSTCODE": "MAPPED_POSTCODE"})
    )

    # Create dataframe of UPRNs with their mapped POSTCODE from the same building where available
    postcodes_df = (
        features_df.select(["UPRN", "ID"])
        .join(id_postcode_mapping_df, on="ID", how="left")
        .rename({"MAPPED_POSTCODE": "POSTCODE"})
    )

    print(
        "Number of UPRNs with POSTCODE after step 1:",
        postcodes_df.filter(pl.col("POSTCODE").is_not_null()).shape[0],
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
            code_point_df[["POSTCODE", "geometry"]],
            how="left",
            max_distance=max_distance_m,  # maximum distance in metres
            distance_col="distance_to_postcode_m",  # distance in metres
        ).drop(columns="index_right")[["UPRN", "POSTCODE", "distance_to_postcode_m"]]
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
