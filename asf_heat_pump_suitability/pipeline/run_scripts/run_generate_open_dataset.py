import polars as pl
from tqdm import tqdm
import argparse
import s3fs
from datetime import datetime
import logging
from asf_heat_pump_suitability.utils import save_utils


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--property_suitability",
        help="S3 URI to heat pump suitability per property parquet",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lsoa_suitability",
        help="S3 URI to heat pump suitability per LSOA parquet",
        type=str,
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

    parser.add_argument(
        "--save_as",
        help="S3 path to save open data to. If unspecified, save with default filename.",
        type=str,
        default=None,
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.info("Loading suitability per property")
    property_df = pl.read_parquet(args.property_suitability)
    logging.info("Loading suitability per LSOA")
    lsoa_df = pl.read_parquet(args.lsoa_suitability)

    dfs = []
    logging.info("Recalculating weights after removing dummies")
    for lsoa in tqdm(property_df["lsoa"].unique()):
        df = property_df.filter(pl.col("lsoa") == lsoa)
        df = df.with_columns(
            pl.when(
                (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= 0.5
            )
            .then(pl.col("proportional_weight") / pl.col("proportional_weight").sum())
            .otherwise(1)
            .alias("use_weight"),
            pl.when(
                (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= 0.5
            )
            .then(True)
            .otherwise(False)
            .alias("scores_weighted"),
        )

        dfs.append(df)

    property_df = pl.concat(dfs)

    logging.info("Removing LSOAs with <15 properties")
    n_properties = lsoa_df.select(["lsoa", "n_properties"])
    property_df = property_df.join(n_properties, on="lsoa", how="left").filter(
        pl.col("n_properties") >= 15
    )

    logging.info("Calculating summary data for each LSOA")
    # Convert columns to booleans for calculating proportions
    property_df = property_df.with_columns(
        pl.when(pl.col("listed_building_grade").is_null())
        .then(False)
        .otherwise(True)
        .alias("listed_building"),
        pl.when(pl.col("in_conservation_area").is_null())
        .then(False)
        .otherwise(True)
        .alias("conservation_area"),
        pl.when(pl.col("property_type") == "Flat, maisonette or apartment")
        .then(True)
        .otherwise(False)
        .alias("flat_maisonette_apartment"),
        pl.when(
            pl.col("CURRENT_ENERGY_RATING").str.to_uppercase().is_in(["A", "B", "C"])
        )
        .then(True)
        .otherwise(False)
        .alias("epc_c_plus"),
    )

    open_df = property_df.group_by("lsoa").agg(
        median_garden_estimate_m2=pl.col("garden_area_m2").median(),
        property_density_km2=(
            pl.col("Property density (households per KM2)") * pl.col("use_weight")
        ).sum()
        / pl.col("use_weight").sum(),
        rural_urban_class=pl.col("ruc_two_fold").min(),
        proportion_in_conservation_area=(
            pl.col("conservation_area") * pl.col("use_weight")
        ).sum()
        / pl.col("use_weight").sum(),
        proportion_listed_building=(
            pl.col("listed_building") * pl.col("use_weight")
        ).sum()
        / pl.col("use_weight").sum(),
        proportion_flats=(
            pl.col("flat_maisonette_apartment") * pl.col("use_weight")
        ).sum()
        / pl.col("use_weight").sum(),
        proportion_epc_c_plus=(pl.col("epc_c_plus") * pl.col("use_weight")).sum()
        / pl.col("use_weight").sum(),
        proportion_off_gas=(pl.col("OFF GAS") * pl.col("use_weight")).sum()
        / pl.col("use_weight").sum(),
    )

    open_df = open_df.join(lsoa_df, how="left", on="lsoa").rename(
        {"scores_weighted": "weights_used"}
    )

    # Save to S3
    if not args.save_as:
        args.save_as = f"s3://nesta-open-data/asf_heat_pump_suitability/{args.year}Q{args.quarter}/{datetime.today().strftime('%Y%m%d')}_{args.year}_Q{args.quarter}_EPC_heat_pump_suitability_per_lsoa"
    logging.info("Saving to S3")
    save_utils.save_parquet_to_s3(open_df, f"{args.save_as}.parquet")
    fs = s3fs.S3FileSystem()
    with fs.open(path=f"{args.save_as}.csv", mode="wb") as f:
        open_df.write_csv(f)
