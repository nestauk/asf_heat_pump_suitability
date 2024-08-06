"""
Script to calculate garden area (m2) where possible for properties in the domestic EPC register.
"""

import argparse
import logging
import pandas as pd
from tqdm import tqdm
import polars as pl
import geopandas as gpd
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    lat_lon,
    land_extent,
    building_footprint,
    garden_size,
)


def argparser() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: object holding argument attributes
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--use_mapping",
        help="Path to existing mapping of land extent files to LAD boundary geometries",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--epc_path",
        help="Path to EPC file with properties to estimate garden size for",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--save_epc_gardens",
        help="Path to save output file with garden size per EPC record to",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--save_land_file_bounds",
        help="Path to save land extent file bounds to",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    _args = argparser()

    # Load EPC x, y coordinates in CRS: EPSG:27700
    epc_gdf = pl.read_parquet(
        _args.epc_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )
    epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_gdf)[["UPRN", "geometry"]]

    if not _args.use_mapping:
        # Get land extent file boundaries
        land_file_bounds = land_extent.generate_gdf_map_file_to_bounds(
            save_as=_args.save_land_file_bounds
        )
    else:
        # Load existing file with land extent files mapped to LAD boundaries
        land_file_bounds = gpd.read_file(_args.use_mapping, crs="EPSG:27700")

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
            )  # TODO is there a cleaner way to generate the file path here without f-string

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
    epc_gardens_df.to_parquet(_args.save_epc_gardens, engine="pyarrow")
