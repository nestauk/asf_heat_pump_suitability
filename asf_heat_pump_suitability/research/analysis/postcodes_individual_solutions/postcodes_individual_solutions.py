"""
Analysis of postcodes most suitable for individual solutions.

Run with: python asf_heat_pump_suitability/research/analysis/postcodes_individual_solutions/postcodes_individual_solutions.py --local_authorities LOCAL_AUTHORITY

where LOCAL_AUTHORITY is the name of the local authority to run the analysis for, e.g. "plymouth" or "glasgow city".
"""

import argparse
import polars as pl
import geopandas as gpd
import pandas as pd

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.cluster import cluster


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

    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability.getters import load_geodata
    from asf_heat_pump_suitability.pipeline.transform import local_authority
    from asf_heat_pump_suitability import config

    args = parse_arguments()
    local_authorities = args.local_authorities
    tolerance_m = config["constant"]["clustering"]["tolerance_m"]

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    print(f"Loading {local_authorities} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        config["output"]["dataset"]["domestic_uprns_with_features"].format(
            local_authority=local_authority_dict["url_slug"]
        )
    )

    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    print("Loading clusters...")
    clusters_gdf = gpd.read_parquet(
        config["output"]["dataset"]["tech_clusters"].format(
            local_authorities=local_authority_dict["url_slug"],
            tolerance_m=tolerance_m,
        ),
    ).to_crs(epsg=27700)

    uprns_df = cluster.map_df_uprns_to_clusters(
        uprns_df=uprns_df, buildings_gdf=buildings_gdf, clusters_gdf=clusters_gdf
    )

    # Drop cluster_ids starting with DESNZ_HNZ
    uprns_df = uprns_df.filter(~uprns_df["cluster_id"].str.startswith("DESNZ_HNZ"))

    # Map each UPRN to its postcode
    print("Mapping UPRNs to postcodes...")
    import boto3

    s3_client = boto3.client("s3")

    path = config["data"]["geodata"]["gb_uprn_country_mapping"]
    bucket_name = path.split("s3://")[1].split("/")[0]
    prefix = path.split(f"s3://{bucket_name}/")[1]

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = [
        f"s3://{bucket_name}/{obj['Key']}"
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]

    uprn_to_country_df = pd.concat(
        [pd.read_csv(file, usecols=["UPRN", "PCDS"]) for file in files],
        ignore_index=True,
    )

    # Number of UPRNs per postcode
    uprn_to_country_df = (
        uprn_to_country_df.groupby("PCDS").agg({"UPRN": "count"}).reset_index()
    )
    uprn_to_country_df.rename(
        columns={"UPRN": "n_uprns", "PCDS": "postcode"}, inplace=True
    )

    # Merge the UPRN to country mapping with the UPRNs DataFrame
    uprns_df = uprns_df.join(
        pl.DataFrame(uprn_to_country_df),
        on="postcode",
        how="left",
    )

    # Identify postcodes most suitable for individual solutions, those with cluster_id starting with "IND"
    uprns_df = uprns_df.with_columns(
        pl.when(uprns_df["cluster_id"].str.startswith("IND"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("suitable_for_individual_solutions")
    )

    # Identify postcodes less suitable for individual solutions, those with cluster_id starting with "NHP", "COM" or "DESNZ"
    uprns_df = uprns_df.with_columns(
        pl.when(
            uprns_df["cluster_id"].str.startswith("NHP")
            | uprns_df["cluster_id"].str.startswith("COM")
            | uprns_df["cluster_id"].str.startswith("DESNZ")
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("less_suitable_for_individual_solutions")
    )

    # Get df of postcodes where less_suitable_for_individual_solutions is True and suitable_for_individual_solutions is False
    postcodes_df = uprns_df.filter(
        (uprns_df["less_suitable_for_individual_solutions"])
        & ~(uprns_df["suitable_for_individual_solutions"])
    ).select(["postcode"])

    # TODO: use all postcodes in the local authority, not just those with UPRNs

    # for each postcode, get the number of UPRNs and the number of UPRNs
    uprns_df.select(
        [
            "postcode",
            "n_uprns",
            "suitable_for_individual_solutions",
            "TNURE",
            "max_contiguous_outdoor_space_area_m2",
        ]
    ).groupby("postcode").agg(
        [
            pl.first("n_uprns").alias("total_uprns"),
            pl.any("suitable_for_individual_solutions").alias(
                "is_suitable_for_individual_solutions"
            ),
            pl.sum(
                pl.when(uprns_df["TNURE"] == "owner-occupied").then(1).otherwise(0)
            ).alias("n_owner-occupied"),
            pl.sum(pl.when(uprns_df["TNURE"].is_null()).then(1).otherwise(0)).alias(
                "n_unknown_tenure"
            ),
            pl.mean("max_contiguous_outdoor_space_area_m2").alias(
                "mean_max_contiguous_outdoor_space_area_m2"
            ),
        ]
    ).with_columns(
        # percent uprns in owner-occupied tenure
        (pl.col("n_owner-occupied") / pl.col("total_uprns") * 100).alias(
            "percent_owner_occupied"
        ),
        # percent uprns with unknown tenure
        (pl.col("n_unknown_tenure") / pl.col("total_uprns") * 100).alias(
            "percent_unknown_tenure"
        ),
    )
