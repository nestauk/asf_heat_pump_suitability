"""
Analysis of postcodes most suitable for individual solutions.

Run with: python asf_heat_pump_suitability/research/analysis/postcodes_individual_solutions/postcodes_individual_solutions.py --local_authority LOCAL_AUTHORITY

where LOCAL_AUTHORITY is the name of the local authority to run the analysis for, e.g. "plymouth" or "east suffolk".
"""

# package imports
import argparse
import polars as pl
import geopandas as gpd
import os
import boto3
from datetime import datetime

# local imports
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.cluster import cluster
from asf_heat_pump_suitability.getters import load_geodata
from asf_heat_pump_suitability.pipeline.transform import local_authority as tla


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authority",
        help="Local authority to run the analysis for, e.g. 'plymouth' or 'glasgow city'.",
        type=str,
        required=True,
    )

    return parser.parse_args()


la_code_mapping = {
    "babergh": "E07000200",
    "wiltshire": "E06000054",
    "east suffolk": "E07000244",
    "ipswich": "E07000202",
    "mid suffolk": "E07000203",
    "west suffolk": "E07000245",
    "breckland": "E07000143",
    "west berkshire": "E06000037",
    "somerset": "E06000066",
    "dorset": "E06000059",
    "plymouth": "E06000026",
}

if __name__ == "__main__":
    args = parse_arguments()
    local_authority = args.local_authority.strip().lower()

    if local_authority not in la_code_mapping:
        raise ValueError(
            f"Local authority '{local_authority}' is not present in la_code_mapping. "
            f"Available options: {list(la_code_mapping.keys())}"
        )

    tolerance_m = config["constant"]["clustering"]["tolerance_m"]
    local_authority_dict = tla.get_dict_la_data(local_authority)

    print(f"Loading {local_authority} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        config["output"]["dataset"]["domestic_uprns_with_features"].format(
            local_authority=local_authority_dict["url_slug"]
        )
    )

    # Load buildings geodataframe for the local authority
    print(f"Loading {local_authority} buildings geodataframe...")
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    # Load clusters geodataframe for the local authority
    print(f"Loading {local_authority} clusters ...")
    clusters_gdf = gpd.read_parquet(
        config["output"]["dataset"]["tech_clusters"].format(
            local_authorities=local_authority_dict["url_slug"],
        ),
    ).to_crs(epsg=27700)

    # Map UPRNs to clusters
    print(f"Mapping {local_authority} UPRNs to clusters...")
    uprns_df = cluster.map_df_uprns_to_clusters(
        uprns_df=uprns_df, buildings_gdf=buildings_gdf, clusters_gdf=clusters_gdf
    )

    # Dropping UPRNs in HN zone
    uprns_df = uprns_df.filter(~pl.col("cluster_id").str.starts_with("DESNZ_HNZ"))

    print("Loading UPRN to postcode mapping from S3...")
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

    uprn_to_postcode_df = pl.concat(
        [pl.read_csv(file, columns=["UPRN", "PCDS", "lad25cd"]) for file in files]
    ).rename({"PCDS": "postcode"})

    # Filter the UPRN to country mapping to only include UPRNs in the local authority
    uprn_to_postcode_df = uprn_to_postcode_df.filter(
        pl.col("lad25cd") == la_code_mapping[local_authority]
    )

    # Extend uprns_df with postcode information
    uprns_df = uprns_df.join(
        uprn_to_postcode_df.select(["UPRN", "postcode"]),
        on="UPRN",
        how="left",
    )

    # Add n_domestic_uprns_in_postcode as a column in uprns_df
    uprns_df = uprns_df.with_columns(
        pl.col("UPRN").count().over("postcode").alias("n_domestic_uprns_in_postcode")
    )

    # Aggregate UPRN counts per postcode
    uprns_per_postcode_df = uprn_to_postcode_df.group_by("postcode").agg(
        pl.col("UPRN").count().alias("n_uprns_in_postcode")
    )

    # Extend uprns_df with overall UPRN counts per postcode
    uprns_df = uprns_df.join(
        uprns_per_postcode_df.select(["postcode", "n_uprns_in_postcode"]),
        on="postcode",
        how="left",
    )

    # Identify suitability flags at UPRN level based on cluster_id prefixes
    # 1) Suitable for individual solutions if cluster_id starts with "IND"
    uprns_df = uprns_df.with_columns(
        pl.col("cluster_id")
        .str.starts_with("IND")
        .alias("suitable_for_individual_solutions")
    )

    # 2) Less suitable for individual solutions if cluster_id starts with "NHP", "COM", or "DESNZ"
    uprns_df = uprns_df.with_columns(
        (
            pl.col("cluster_id").str.starts_with("NHP")
            | pl.col("cluster_id").str.starts_with("COM")
            | pl.col("cluster_id").str.starts_with("DESNZ")
        ).alias("less_suitable_for_individual_solutions")
    )

    # Suitability at postcode level
    # Less suitable for Individual solutions at postcode level: if less_suitable_for_individual_solutions is True and suitable_for_individual_solutions is False
    # Essentially, if there is at least one UPRN suitable for individual solutions in the postcode, then the postcode is considered suitable for individual solutions.
    postcodes_less_suitable_individual_df = (
        uprns_df.group_by("postcode")
        .agg(
            (
                pl.col("less_suitable_for_individual_solutions").any()
                & ~pl.col("suitable_for_individual_solutions").any()
            ).alias("postcode_less_suitable_for_individual_solutions")
        )
        .filter(pl.col("postcode_less_suitable_for_individual_solutions"))
        .select("postcode")
    )

    # Save postcodes_df locally as a CSV file
    postcodes_less_suitable_individual_df.write_csv(
        os.path.join(
            PROJECT_DIR,
            "outputs",
            f"postcodes_{local_authority}_less_suitable_for_individual_solutions_{datetime.today().strftime('%Y%m%d')}.csv",
        )
    )

    # Group and generate summary df
    summary_per_postcode_df = (
        uprns_df.group_by("postcode")
        .agg(
            [
                pl.col("n_domestic_uprns_in_postcode").first(),
                pl.col("n_uprns_in_postcode").first(),
                pl.col("suitable_for_individual_solutions")
                .any()
                .alias("one_uprn_or_more_suitable_for_individual_solutions"),
                pl.col("less_suitable_for_individual_solutions")
                .any()
                .alias("one_uprn_or_more_less_suitable_for_individual_solutions"),
                pl.col("suitable_for_individual_solutions")
                .count()
                .alias("n_suitable_for_individual_solutions"),
                pl.col("less_suitable_for_individual_solutions")
                .count()
                .alias("n_less_suitable_for_individual_solutions"),
                (
                    pl.col("n_suitable_for_individual_solutions")
                    / pl.col("n_domestic_uprns_in_postcode")
                    * 100
                ).alias("percent_suitable_for_individual_solutions"),
                (
                    pl.col("n_less_suitable_for_individual_solutions")
                    / pl.col("n_domestic_uprns_in_postcode")
                    * 100
                ).alias("percent_less_suitable_for_individual_solutions"),
                pl.col("TENURE").eq("owner-occupied").sum().alias("n_owner_occupied"),
                pl.col("TENURE").is_null().sum().alias("n_unknown_tenure"),
                pl.col("max_contiguous_outdoor_space_area_m2")
                .mean()
                .alias("mean_max_contiguous_outdoor_space_area_m2"),
            ]
        )
        .with_columns(
            [
                # Percent UPRNs in owner-occupied tenure
                (
                    pl.col("n_owner_occupied")
                    / pl.col("n_domestic_uprns_in_postcode")
                    * 100
                ).alias("percent_domestic_owner_occupied"),
                # Percent UPRNs with unknown tenure
                (
                    pl.col("n_unknown_tenure")
                    / pl.col("n_domestic_uprns_in_postcode")
                    * 100
                ).alias("percent_unknown_tenure"),
                # Postcode suitable for individual solutions: True if at least one UPRN is suitable for individual solutions, False otherwise
                pl.col("one_uprn_or_more_suitable_for_individual_solutions").alias(
                    "postcode_suitable_for_individual_solutions"
                ),
                # Postcode less suitable for individual solutions: True if at least one UPRN is less suitable for individual solutions and no UPRNs are suitable for individual solutions, False otherwise
                (
                    ~pl.col("one_uprn_or_more_suitable_for_individual_solutions")
                    & pl.col("one_uprn_or_more_less_suitable_for_individual_solutions")
                ).alias("postcode_less_suitable_for_individual_solutions"),
            ]
        )
        .drop(
            [
                "one_uprn_or_more_suitable_for_individual_solutions",
                "one_uprn_or_more_less_suitable_for_individual_solutions",
            ]
        )
    )

    # Save summary per postcode df locally as a CSV file
    summary_per_postcode_df.write_csv(
        os.path.join(
            PROJECT_DIR,
            "outputs",
            f"summary_{local_authority}_postcode_suitability_individual_solutions_{datetime.today().strftime('%Y%m%d')}.csv",
        )
    )

    # Check that postcodes in postcodes_less_suitable_individual_df are the postcodes in summary_per_postcode_df where postcode_less_suitable_for_individual_solutions is True
    postcodes_less_suitable_individual_df_check = summary_per_postcode_df.filter(
        pl.col("postcode_less_suitable_for_individual_solutions")
    ).select("postcode")

    assert postcodes_less_suitable_individual_df.frame_equal(
        postcodes_less_suitable_individual_df_check
    ), "Postcodes in postcodes_less_suitable_individual_df do not match postcodes in summary_per_postcode_df where postcode_less_suitable_for_individual_solutions is True"
