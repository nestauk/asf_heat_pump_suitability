"""
Create geojson files to use as inputs to tippecanoe map tiler for the heatpump suitability webmap.

This script gets the relevant England & Wales LSOA and Scotland DataZone boundaries from the /mapdata/ prefix in
the heat pump suitability s3 bucket. Concatenates the geometries. Joins the suitability scoring (provided via config or as a flag)
to the merged geometries and writes the output back to s3 (as default) or a local file.

We're using four levels of generalisation for the geojsons, to reflect the level of detail needed at different
zooms in the webmap. For England and Wales these are the 3 levels of generalisation officially published and an
additional ultra-generalised version that we compute. As Scotland doesn't publish official generalisations we
simplify the Scottish DataZones to reflect the generalisation seen in ONS geography data. Simplification is
conducted using geopandas, which implements the shapely/GEOS implementation of the topology preserving simplifier
(similar in principle to the Douglas-Peucker algorithm).

Detailed geometries are the full available resolution of the boundaries, generalised have a tolerance of 20m, super
generalised 200m and ultra generalised 500m. These tolerances act to simplify geometries by removing intermediary
vertices that fall within the tolerance, larger tolerances in general producing simpler geometries with fewer vertices.

The s3 locations of the necessary geojsons are specified in config/base.yml and can be updated there if required.

All flag are optional and can be used as follows:

The --scores or -s flag allows you to pass the full s3 uri or local filepath to sutiability scores data. The default
approach is to try to find the latest available scores data in the heat pump suitability s3 bucket.

The --date_outputs or -d flag can be used to prepend the date to the output file in the form YYYYMMDD_ the default is
False (no date prepended).

The --output_dir or -o flag can be used to specify the output location as either a local directory or s3 prefix. The
default is to use the directory listed in the config 'base.yaml' under 'webmap_data_source; out_key'.

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_make_webmap_geojsons.py
"""

import pandas as pd
import geopandas as gpd
import os
import sys
import argparse
import logging
from argparse import ArgumentParser
from datetime import datetime

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import webmap_utils

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-s",
        "--scores",
        help="s3 uri for required suitability scores per lsoa parquet file, defaults to latest available.",
        type=str,
        default="LATEST",
    )

    parser.add_argument(
        "-d",
        "--date_outputs",
        help="whether to prepend the date to the output file name, defaults to False.",
        action="store_true", 
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="output directory or s3 prefix, defaults to config base.yaml if not stated.",
        type=str,
        default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Get Scores file uri logic.
    if args.scores == "LATEST":
        # Get latest version of suitability scores.
        logger.info(f"Getting latest scores file.")
        file_uri = webmap_utils.get_str_latest_scores_parquet_file_uri()
    else:
        logger.info(f"Checking user provided scores file path.")
        file_uri = webmap_utils.check_exists_str_file_uri(args.scores)

    # check parquet - probably assuming a single-part parquet file (e.g. non-spark-like)
    filename, file_extension = os.path.splitext(file_uri)
    if file_extension not in ['.pq', '.parquet', '.pqt']:
        raise NotImplementedError(f"Scores file must be in parquet format.")

    if args.output_dir:
        # If an output directory has been specified, check it exists.
        logger.info(f"Checking user provided output directory.")
        output_dir = webmap_utils.check_exists_str_output_directory(args.output_dir)
    else:
        output_dir = config['webmap_data_source']['out_key']
        logger.info(f"Using output directory from config: {output_dir}")

    # Load scores file
    logger.info(f"Reading suitability scores from {file_uri}")
    scores = pd.read_parquet(file_uri)

    # Now iterate over the different resolution geojsons to create the joined files.
    geojson_filestrings = config['webmap_data_source']
    key_order = (('EW_detailed_lsoa_geojson', 'S_detailed_datazone_geojson'),
                 ('EW_generalised_lsoa_geojson', 'S_generalised_datazone_geojson'),
                 ('EW_super_generalised_lsoa_geojson', 'S_super_generalised_datazone_geojson'),
                 ('EW_ultra_generalised_lsoa_geojson', 'S_ultra_generalised_datazone_geojson'))
    outfile_name = ['detailed_areas', 'generalised_areas', 'super_generalised_areas', 'ultra_generalised_areas']
    logger.info(f"Creating output geojsons.")
    for i, (ew, s) in enumerate(key_order):
        # Load geometries
        ew_gdf = gpd.read_file(geojson_filestrings[ew]).loc[:, ['LSOA21CD', 'geometry']].rename(columns={'LSOA21CD': 'area_code'})
        s_gdf = gpd.read_file(geojson_filestrings[s]).loc[:, ['DataZone', 'geometry']].rename(columns={'DataZone': 'area_code'})

        # Merge England and Wales LSOAs with Scottish DataZones.
        ews_gdf = pd.concat([ew_gdf, s_gdf], ignore_index=True)
        del ew_gdf, s_gdf

        # Merge relevant heat pump suitability data
        ews_gdf = ews_gdf.merge(
            scores[['lsoa', 'ASHP_S_avg_score_weighted', 'ASHP_N_avg_score_weighted',
                    'GSHP_S_avg_score_weighted', 'GSHP_N_avg_score_weighted',
                    'SGL_S_avg_score_weighted', 'SGL_N_avg_score_weighted',
                    'HN_S_avg_score_weighted', 'HN_N_avg_score_weighted']],
            left_on='area_code',
            right_on='lsoa')
        # Save to file
        output_dir = output_dir + "/" if output_dir[-1] != "/" else output_dir
        if args.date_outputs:
            outfile = f"{output_dir}{datetime.today().strftime('%Y%m%d')}_{outfile_name[i]}.geojson"
        else:
            outfile = f"{output_dir}{outfile_name[i]}.geojson"
        ews_gdf[['area_code', 'ASHP_S_avg_score_weighted', 'ASHP_N_avg_score_weighted',
                 'GSHP_S_avg_score_weighted', 'GSHP_N_avg_score_weighted',
                 'SGL_S_avg_score_weighted', 'SGL_N_avg_score_weighted',
                 'HN_S_avg_score_weighted', 'HN_N_avg_score_weighted', 'geometry']].to_file(outfile)
        logger.info(f"Geojson output created for {outfile_name[i]}")
