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
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.pipeline.suitability import calculate_suitability


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

    logging.info("Loading EPC data with features")
    epc_df = pl.read_parquet(path=args.features)
    logging.info("Loading garden size estimates")
    gardens = pl.read_parquet(path=args.gardens)
    logging.info("Loading weights")
    weights = pl.read_parquet(path=args.weights)

    logging.info("Joining EPC features data with garden size estimates and weights")
    epc_df = epc_df.join(gardens, how="left", on="UPRN")
    epc_df = epc_df.join(weights, how="left", on="UPRN")

    logging.info(f"Saving augmented EPC data")
    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}{q}/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_epc_augmented.parquet"
    save_utils.save_to_s3(epc_df, save_as)

    epc_df = epc_df.with_columns(
        pl.col("garden_area_m2")
        .fill_null(pl.col("msoa_avg_outdoor_space_m2"))
        .alias("garden_area_m2")
    ).drop("msoa_avg_outdoor_space_m2")

    logging.info("Filtering EPC data to rows with n_features >= minimum threshold")
    features = [
        "CURRENT_ENERGY_RATING",
        "property_type",
        "ruc_two_fold",
        "off_gas",
        "listed_building",
        "in_protected_area",
        "garden_area_m2",
        "households_per_km2",
        "has_anchor_properties",
        "heatpump_installation_percentage",
    ]
    epc_df = calculate_suitability.filter_df_minimum_features(epc_df, features=features)

    tech_types = [
        "ASHP_S",
        "ASHP_N",
        "GSHP_S",
        "GSHP_N",
        "SGL_S",
        "SGL_N",
        "HN_S",
        "HN_N",
    ]

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

    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}{q}/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_heat_pump_suitability_per_property.parquet"
    save_utils.save_to_s3(epc_df, save_as)

    logging.info("Weighting scores and aggregating per LSOA")
    weighted_scores = []
    for lsoa_code in tqdm(epc_df["lsoa"].unique()):
        lsoa_df = epc_df.filter(pl.col("lsoa") == lsoa_code)
        lsoa_df = calculate_suitability.compute_df_weighted_score(lsoa_df)
        weighted_scores.append(
            calculate_suitability.compute_dict_lsoa_suitability_scores(
                lsoa_df, lsoa_code
            )
        )
    # Must have at least 15 properties to be included in final dataset
    suitability_df = pl.DataFrame(weighted_scores).filter(pl.col("n_properties") >= 15)
    suitability_df = suitability_df.with_columns(pl.col(pl.Float64).round(3))

    # TODO add Data Zone names
    logging.info("Get LSOA names and join to suitability dataset")
    lsoa_names_df = pl.read_csv(
        config["data_source"]["EW_ons_lsoa_lad_lookup"],
        columns=["LSOA21CD", "LSOA21NM"],
    )
    suitability_df = suitability_df.join(
        lsoa_names_df, left_on="lsoa", right_on="LSOA21CD", how="left"
    ).rename({"LSOA21NM": "lsoa_name"})

    logging.info("Saving LSOA heat pump suitability scores")
    save_as = f"s3://asf-heat-pump-suitability/outputs/{y}{q}/{datetime.today().strftime('%Y%m%d')}_{y}_Q{q}_heat_pump_suitability_per_lsoa"
    save_utils.save_to_s3(suitability_df, f"{save_as}.parquet")
    save_utils.save_to_s3(suitability_df, f"{save_as}.csv")
