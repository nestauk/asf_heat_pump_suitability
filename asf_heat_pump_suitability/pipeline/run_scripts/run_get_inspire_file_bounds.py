"""
Generate bounding polygons of each INSPIRE land registry file and save to S3.

Land registry files are partitioned by Registration County in Scotland and by local authority in England and Wales. To
match land registry files with Microsoft building footprint files, we first must identify the bounding geometry of each
land registry file.

Scotland INSPIRE files are smaller (they are shapefiles) and there are fewer (33). They are loaded individually in this
script and the bounding polygon generated.

England and Wales INSPIRE files are larger (.gml files) and there are more (300+). They are matched to existing local
authority boundaries via string matching based on their filenames. Any nulls are filled by loading the land registry
file and identifying which local authority boundary it should be matched to based on the geometry.

To run:
python asf_heat_pump_suitability/pipeline/run_scripts/run_get_inspire_file_bounds.py -y [YYYY] -q [N] -n all

[Set -n nation flag to "ew" or "s" for generating file bounds either England and Wales or Scotland INSPIRE files only.]
"""

import argparse
from argparse import ArgumentParser
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.prepare_features import (
    land_extent,
)


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = ArgumentParser()

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
        help="Nations to get INSPIRE land registry file bounds for, out of England and Wales; Scotland; or all.",
        type=str,
        choices=["ew", "s", "all"],
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    year = args.year
    q = args.quarter

    if args.nations in ["ew", "all"]:
        gdf = land_extent.generate_gdf_file_bounds_ew(
            path=config["data_source"]["EW_inspire_land_extent_dir"]
        )
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{year}_inspire_file_bounds_EW.geojson"
        gdf.to_file(save_as)

    if args.nations in ["s", "all"]:
        gdf = land_extent.generate_gdf_file_bounds_s(
            config["data_source"]["S_inspire_land_extent_dir"]
        )
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{year}_inspire_file_bounds_S.geojson"
        gdf.to_file(save_as)
