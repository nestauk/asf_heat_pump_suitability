"""
Calculate suitability scores of different low-carbon heating technologies in Nesta and 'conventional' views for individual
properties and LSOAs.

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_suitability.py --weights [path/to/weighted/EPC] --features [path/to/EPC/with/features] --gardens [path/to/garden/size/estimates] -y [YYYY] -q [Q]

NB: this pipeline takes the outputs from the following scripts as inputs:
- asf_heat_pump_suitability/pipeline/run_scripts/run_compute_epc_weights.py
- asf_heat_pump_suitability/pipeline/run_scripts/run_add_features.py
- asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_garden_size.py
"""

import polars as pl
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.pipeline.prepare_features import (
    property_type,
    output_areas,
)
from asf_heat_pump_suitability.pipeline.suitability import (
    calculate_suitability,
)


def parse_arguments():
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        help="Path to weighted EPC data, the output of `run_compute_epc_weights.py`",
        required=True,
    )

    parser.add_argument(
        "--features",
        help="Path to EPC data with added features, the output of `run_add_features.py`",
        required=True,
    )

    parser.add_argument(
        "--gardens",
        help="Path to deduplicated estimated garden size data, the output of `run_calculate_garden_size.py`.",
        required=True,
    )

    parser.add_argument(
        "-y",
        "--year",
        help="EPC data year. Format YYYY",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-q",
        "--quarter",
        help="EPC data quarter",
        type=int,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    y = args.year
    q = args.quarter
    features = config["features"]
    tech_types = config["tech_types"]

    logging.info("Loading EPC data with features")
    epc_df = pl.read_parquet(args.features)
    logging.info("Loading garden size estimates")
    gardens = pl.read_parquet(args.gardens)
    logging.info("Loading weights")
    weights = pl.read_parquet(args.weights)

    logging.info("Joining EPC features data with garden size estimates and weights")
    epc_df = epc_df.join(gardens, how="left", on="UPRN")
    epc_df = epc_df.join(weights, how="left", on="UPRN")

    logging.info(f"Saving augmented EPC data")
    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}Q{q}/augmented_epc/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_epc_augmented.parquet"
    save_utils.save_to_s3(epc_df, save_as)

    epc_df = epc_df.with_columns(
        pl.col("garden_area_m2")
        .fill_null(pl.col("msoa_avg_outdoor_space_m2"))
        .alias("garden_area_m2")
    ).drop("msoa_avg_outdoor_space_m2")

    logging.info("Filtering EPC data to rows with n_features >= minimum threshold")
    epc_df = calculate_suitability.filter_df_minimum_features(epc_df, features=features)

    scores = []
    for tech_type in tech_types:
        logging.info(f"Calculating suitability scores for tech type: {tech_type}")
        epc_scores_df = calculate_suitability.compute_df_avg_score_per_epc(
            epc_df, tech_type
        )
        scores.append(epc_scores_df)

    logging.info("Joining all scores to EPC dataset")
    for score_df in scores:
        epc_df = epc_df.join(score_df, on="UPRN", how="left")

    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}Q{q}/suitability/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_heat_pump_suitability_per_property.parquet"
    save_utils.save_to_s3(epc_df, save_as)

    logging.info("Weighting scores and aggregating per LSOA")
    use_cols = (
        ["lsoa", "proportional_weight"]
        + [col for col in epc_df.columns if "score" in col]
        + features
    )
    epc_df = epc_df.select(use_cols)

    weighted_scores = []
    for lsoa_code in tqdm(epc_df["lsoa"].drop_nulls().unique()):
        weighted_scores.append(
            calculate_suitability.aggregate_dict_lsoa_suitability_and_features(
                epc_df, lsoa_code
            )
        )

    logging.info(
        "Filtering to LSOAs with data for at least 15 properties to be included in final dataset"
    )
    suitability_df = pl.DataFrame(weighted_scores).filter(pl.col("n_properties") >= 15)
    suitability_df = suitability_df.with_columns(pl.col(pl.Float64).round(3))

    logging.info("Getting proportion of flats in each LSOA from the census data")
    proportion_flats_df = (
        property_type.transform_df_proportion_census_property_types()
        .filter(pl.col("property_type") == "Flat, maisonette or apartment")
        .select(["lsoa", "census_proportion"])
        .rename({"census_proportion": "census_proportion_flats"})
    )

    logging.info("Getting LSOA & DZ names")
    lsoa_names_df = output_areas.load_df_lsoa_dz_codes_names()

    logging.info(
        "Joining proportion of flats and LSOA & DZ names to suitability dataset"
    )
    suitability_df = suitability_df.join(
        proportion_flats_df, how="left", on="lsoa"
    ).join(lsoa_names_df, left_on="lsoa", right_on="lsoa_code", how="left")

    logging.info("Saving LSOA heat pump suitability scores")
    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}Q{q}/suitability/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_heat_pump_suitability_per_lsoa"
    save_utils.save_to_s3(suitability_df, f"{save_as}.parquet")
    save_utils.save_to_s3(suitability_df, f"{save_as}.csv")

    logging.info("Saving open dataset to nesta-open-data S3 bucket")
    save_as = f"s3://nesta-open-data/asf_heat_pump_suitability/{args.year}Q{args.quarter}/{datetime.today().strftime('%Y%m%d')}_{args.year}_Q{args.quarter}_EPC_heat_pump_suitability_per_lsoa"
    save_utils.save_to_s3(suitability_df, f"{save_as}.parquet")
    save_utils.save_to_s3(suitability_df, f"{save_as}.csv")
