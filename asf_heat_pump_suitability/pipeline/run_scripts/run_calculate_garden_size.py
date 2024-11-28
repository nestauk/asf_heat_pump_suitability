"""
Calculate garden area (m2) where possible for properties in the domestic EPC register using Land Registry data and
Microsoft Building Footprints data.

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_garden_size.py --epc_path [path/to/EPC/data] -y [YYYY] -q [N] -n all

[Set -n nation flag to "ew" or "s" for generating garden size estimates for either England and Wales or Scotland INSPIRE
files only. It is recommended to process England-Wales and Scotland separately due to long run time (2+ days).]
"""

import argparse
import logging
import pandas as pd
from tqdm import tqdm
import polars as pl
import geopandas as gpd
from datetime import datetime
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    lat_lon,
    land_extent,
    building_footprint,
    garden_size,
)


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--epc_path",
        help="Path to EPC file with properties to estimate garden size for. Must have UPRN and x and y coordinate columns.",
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
        "-n",
        "--nations",
        help="Nations to get INSPIRE land registry file bounds for. Of England and Wales (ew); Scotland (s); or all (ews).",
        type=str,
        choices=["ew", "s", "ews"],
        required=True,
    )

    parser.add_argument(
        "--save_as",
        help="Path to save output file with garden size per EPC record to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    year = args.year
    q = args.quarter

    # Load EPC x, y coordinates in CRS: EPSG:27700
    epc_gdf = pl.read_parquet(
        args.epc_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )
    epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_gdf)[["UPRN", "geometry"]]

    # Load land registry and building footprint boundaries
    land_file_bounds = gpd.read_file(
        f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/inspire_file_bounds_{args.nations.upper()}.geojson"
    )
    microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

    # Match land extent files with overlapping building footprint files
    file_matches = garden_size.match_series_files_land_building(
        land_files_gdf=land_file_bounds, building_files_gdf=microsoft_file_bounds
    )

    epc_gardens = []
    prev = None
    total_gardens = 0
    for land_file, building_file in tqdm(file_matches.items()):

        # Only load land extent gdf if we haven't loaded already
        if land_file != prev:
            # Prepare land parcel data
            land_parcels_gdf = land_extent.transform_gdf_land_parcels(
                f"s3://{land_file}"
            )

        # Prepare building footprints data
        building_footprints_gdf = building_footprint.transform_gdf_building_footprints(
            building_file
        )

        # Get intersection of building footprint polygons and land polygons
        intersection_gdf = garden_size.generate_gdf_land_building_overlay(
            land_parcels_gdf=land_parcels_gdf,
            building_footprints_gdf=building_footprints_gdf,
        )

        # Get garden size
        gardens_gdf = garden_size.generate_gdf_garden_size(
            intersection_gdf, land_parcels_gdf
        )
        gardens_gdf = gardens_gdf.assign(
            inspire_land_extent_file=land_file,
            microsoft_building_footprint_file=building_file,
        )

        # Match EPC UPRNs with land parcels and gardens using UPRN coordinates
        # This will keep only EPC records for which garden size can be estimated
        epc_df = gpd.sjoin(
            epc_gdf,
            gardens_gdf,
            how="inner",
            predicate="intersects",
        ).drop(columns=["geometry", "index_right"])

        epc_gardens.append(epc_df)

        # Set prev
        prev = land_file
        total_gardens += len(epc_df)
        logging.info(
            f"Garden size calculated for {total_gardens} EPC properties in total."
        )

    # Get df of all EPC records with garden size estimates
    epc_gardens_df = pd.concat(epc_gardens, ignore_index=True)
    if not args.save_as:
        args.save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_garden_size_estimates_{args.nations.upper()}.parquet"
    epc_gardens_df.to_parquet(args.save_as, engine="pyarrow")
