import polars as pl
import geopandas as gpd
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    land_extent,
    building_footprint,
    garden_size,
)


def argparser():
    """
    Create ArgumentParser and passes arguments to `main()` and runs `main()`.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--use_mapping",
        help="Path to existing land extent mapping with land extent files mapped to "
        "LAD boundaries, and generate bboxes for building footprint files",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--epc_path",
        help="Path to EPC file",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argparser()

    # Load EPC lat/lon
    latlon_in_epc = (
        pl.read_parquet(args.epc_path, columns=["latlon"]).select("").to_list()
    )

    if not args.use_mapping:
        # Get land extent file boundaries
        land_file_bounds = land_extent.generate_gdf_map_file_to_bounds()
    else:
        # load files
        land_file_bounds = gpd.read_file(args.use_mapping, crs="EPSG:27700")

    # Get building footprint file boundaries
    microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

    # Check where building footprint files and land extent files overlap
    file_matches = garden_size.match_dict_files_inspire_microsoft(
        land_files_gdf=land_file_bounds, microsoft_files_gdf=microsoft_file_bounds
    )

    prev = None
    gardens_gdfs = []
    for i, inspire_file, ms_file in enumerate(file_matches.items()):
        # Only load INSPIRE gdf if we haven't loaded already
        if inspire_file != prev:
            # Prepare land parcel data
            land_parcels = land_extent.transform_gdf_land_parcels(
                f"s3://{inspire_file}"
            )  # TODO is there a cleaner way to generate the file path here

        # Prepare building footprints data
        building_footprints = building_footprint.transform_gdf_building_footprints(
            ms_file
        )

        # Get intersection
        intersection = garden_size.generate_gdf_land_building_overlay(
            land_parcels=land_parcels, building_footprints=building_footprints
        )

        # Get garden size
        gardens = garden_size.generate_gdf_garden_size(intersection, land_parcels)
        gardens = gardens.assign(
            inspire_land_file=inspire_file, microsoft_building_footprint_file=ms_file
        )

        # Filter to gardens for EPC properties only
        gardens = gardens.filter(pl.col(""))
        gardens_gdfs.append(gardens)
        # TODO: merge with EPC

        # Set prev
        prev = inspire_file
