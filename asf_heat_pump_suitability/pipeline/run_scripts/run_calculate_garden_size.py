"""
Calculate garden area (m2) where possible for properties in the domestic EPC register using Land Registry data and
Microsoft Building Footprints data.

To run:
python asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_garden_size.py --epc_path [path/to/EPC/data] -y [YYYY] -q [N] --use_mapping [path/to/land/extent/file/boundaries]
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
        "--save_epc_gardens",
        help="Path to save output file with garden size per EPC record to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--use_mapping",
        help="Path to existing mapping of land extent files to council/LAD boundary geometries. Recommended if available.",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--save_land_file_bounds",
        help="Path to save land extent file bounds to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    save_land_file_bounds = args.save_land_file_bounds
    save_epc_gardens = args.save_epc_gardens
    year = args.year
    q = args.quarter

    # Load EPC x, y coordinates in CRS: EPSG:27700
    epc_gdf = pl.read_parquet(
        args.epc_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )
    epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_gdf)[["UPRN", "geometry"]]

    if not args.use_mapping:
        if not save_land_file_bounds:
            save_land_file_bounds = f"s3://asf-heat-pump-suitability/outputs/{year}_land_parcels_with_file_polygons.geojson"
        # Get land extent file boundaries
        land_file_bounds = land_extent.generate_gdf_map_file_to_bounds(
            save_as=save_land_file_bounds
        )
    else:
        # Load existing file with land extent files mapped to LAD boundaries
        land_file_bounds = gpd.read_file(args.use_mapping)

    # Get building footprint file boundaries
    microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

    # Check where building footprint files and land extent files overlap
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
    if not save_epc_gardens:
        save_epc_gardens = f"s3://asf-heat-pump-suitability/outputs/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_garden_size_estimates.parquet"
    epc_gardens_df.to_parquet(save_epc_gardens, engine="pyarrow")
