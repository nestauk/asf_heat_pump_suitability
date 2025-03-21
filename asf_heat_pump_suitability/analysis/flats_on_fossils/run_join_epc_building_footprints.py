import polars as pl
import logging
import shapely
import argparse
from tqdm import tqdm
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import (
    building_footprint,
    lat_lon,
)


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epc",
        help="Path to processed and deduplicated EPC dataset in parquet file format",
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
        help="Path to save output file with building footprint information per EPC record to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.info("Loading EPC UPRNs")
    epc_df = pl.read_parquet(args.epc, columns=["UPRN"])

    logging.info("Adding lat/lon data to EPC")
    uprn_coords_df = lat_lon.transform_df_osopen_uprn_latlon()
    epc_df = epc_df.join(uprn_coords_df, how="left", on="UPRN")
    epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_df, usecols=["UPRN"])[
        ["UPRN", "geometry"]
    ]

    logging.info("Loading UK UPRNs")
    uk_uprns_gdf = get_datasets.get_df_osopen_uprn_latlon()
    uk_uprns_gdf = lat_lon.generate_gdf_uprn_coords(
        uk_uprns_gdf, usecols=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )[["UPRN", "geometry"]]

    microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

    logging.info("Getting list of UPRNs for each building footprint file")
    uprns_per_file = microsoft_file_bounds.sjoin(
        epc_gdf, how="inner", predicate="intersects"
    )
    uprns_per_file = uprns_per_file.groupby("ms_url")["UPRN"].agg(list).to_dict()

    epc_footprint_dfs = []

    logging.info("Joining EPC data to building footprint data")
    for building_file, uprns in tqdm(uprns_per_file.items()):
        # Prepare building footprints data
        try:
            building_footprints_gdf = (
                building_footprint.transform_gdf_building_footprints(building_file)
            )
        except shapely.errors.GEOSException as e:
            logging.warning(
                f"Error loading building footprint file {building_file}. Error message: {e}.\n"
                f"Skipping this land extent & building footprint pairing."
            )
            continue

        uprn_gdf = epc_gdf.loc[epc_gdf["UPRN"].isin(uprns)]
        uprns_per_building_df = (
            building_footprints_gdf[["building_id", "geometry"]]
            .sjoin(uk_uprns_gdf, how="left", predicate="contains")
            .drop(columns=["index_right"])
            .groupby("building_id")
            .agg({"UPRN": "count"})
            .rename(columns={"UPRN": "UPRN_count_per_building"})
        )

        epc_footprint_dfs.append(
            pl.from_pandas(
                building_footprints_gdf.sjoin(
                    uprn_gdf, how="inner", predicate="contains"
                )
                .drop(columns=["index_right", "geometry"])
                .join(uprns_per_building_df, how="left", on="building_id")
            )
        )

    epc_footprint_df = pl.concat(epc_footprint_dfs)

    if not args.save_as:
        args.save_as = f"s3://asf-heat-pump-suitability/outputs/{args.year}Q{args.quarter}/analysis/{args.year}_Q{args.quarter}_epc_building_footprints.parquet"

    save_utils.save_to_s3(epc_footprint_df, args.save_as)
