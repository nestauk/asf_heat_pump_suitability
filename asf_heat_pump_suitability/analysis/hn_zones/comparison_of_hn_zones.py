"""
This script analyses DESNZ heat network zones and Nesta's heat pump suitability data
for one or more Local Authorities (LAs). It performs spatial joins, computes coverage
fractions, calculates various statistics, and exports the results.

**Key Steps**:
1. **Spatial Analysis**:
   - Loads DESNZ heat network zones (GeoPackage) and LSOA polygons (Shapefile), ensuring consistent CRS.
   - Performs spatial intersections to calculate the fraction of each LSOA covered by heat network zones.

2. **Nesta Data Processing**:
   - Loads and filters Nesta heat pump suitability scores for each LA.
   - Identifies LSOAs outside the target LA(s).

3. **Statistical Metrics**:
   - Calculates average suitability scores for LSOAs that are inside vs. outside DESNZ pilot areas.
   - Computes Mean Absolute Error (MAE) comparing DESNZ coverage fraction and Nesta’s heat network score.

4. **Data Export**:
   - For each LA, outputs:
       - A GeoPackage (`la_name_with_desnz_hn_lsoa.gpkg`)
       - A JSON file listing LSOAs (`la_name_hp_suitability_lsoas.json`)
       - A Parquet file containing final suitability scores with DESNZ coverage (`la_name_hp_suitability_scores_with_desnz.parquet`)
       - A CSV of those same scores (`la_name_hp_suitability_scores_with_desnz.csv`)
   - Also creates a combined CSV (`la_mae_data.csv`) aggregating MAE metrics across all LAs.
   - Logs all steps and statistics (`script_output.log` by default).

**Note**: For a “region” such as **Greater Manchester**, this script further processes
each sub-LA (Bolton, Bury, Manchester, etc.) within that region using the same underlying
GeoPackage.

**Outputs** (per LA):
- GeoPackage: `<la_name>_with_desnz_hn_lsoa.gpkg`
- JSON: `<la_name>_hp_suitability_lsoas.json`
- Parquet: `<la_name>_hp_suitability_scores_with_desnz.parquet`
- CSV: `<la_name>_hp_suitability_scores_with_desnz.csv`
- Combined MAE CSV for all LAs: `la_mae_data.csv`
- Log: `script_output.log`

**How to Run the Script**:
To run the script, use the following command:

python comparison_of_hn_zones.py [--optional_threshold OPTIONAL_THRESHOLD] [--read_in_s3] [--save_to_s3]

Example:
    # Run the script with default settings (local files, no threshold)
    python comparison_of_hn_zones.py

    # Run the script with a specific threshold
    python comparison_of_hn_zones.py --optional_threshold 0.1

    # Run the script with files from S3
    python comparison_of_hn_zones.py --read_in_s3

    # Run the script and save outputs to S3
    python comparison_of_hn_zones.py --save_to_s3

    # Run the script with a specific threshold, read from S3, and save to S3
    python comparison_of_hn_zones.py --optional_threshold 0.1 --read_in_s3 --save_to_s3
"""

import logging
import json
import os
import geopandas as gpd
from typing import List, Dict
from asf_heat_pump_suitability import PROJECT_DIR
from utils.log_save_utils import (
    setup_logging_and_file_path,
    setup_paths,
    optionally_upload_file_to_s3,
    save_gdf_to_gpkg,
)
from utils.spatial_utils import load_transform_hn_geodata
from utils.nesta_hp_suitability_utils import filter_la_nesta_hp_scores
from utils.hnz_comparison_analysis_utils import (
    add_DESNZ_pilot_fraction,
    calculate_average_scores_for_thresholds,
    calculate_mae_for_all,
    calculate_mae_for_pilot_score,
)
import argparse
import polars as pl
from config.hnz_config import (
    S3_BUCKET,
    S3_KEY_DIR,
    LOCAL_AUTHORITIES,
    DEFAULT_THRESHOLD,
    THRESHOLDS,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# 1. Main Processing Functions
# ---------------------------------------------------------------------------
def process_LA(
    la_name: str,
    gpkg_path: str,
    lsoa_shp_path: str,
    nesta_parquet_path: str,
    output_dir: str,
    optional_threshold: float,
    save_to_s3: bool,
    s3_bucket: str,
    s3_key_dir: str,
    la_mae_data: list,
):
    """
    Processes a local authority (LA). If the LA is actually a region
    like Greater Manchester, it processes each sub-LA with the same dataset.
    Appends mean absolute errors, average heat network scores and missing LSOA information to the list 'la_mae_data'.

    Args:
        la_name (str): Name of the Local Authority.
        gpkg_path (str): Path to the LA (or region) GPKG file (local or S3).
        lsoa_shp_path (str): Path (local or S3) to the LSOA shapefile.
        nesta_parquet_path (str): Path (local or S3) to Nesta's suitability parquet.
        output_dir (str): Local output directory for saving results.
        optional_threshold (float): Threshold for DESNZ pilot fraction (0-1).
        save_to_s3 (bool): Whether to upload outputs to S3.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): S3 path prefix/folder.
        la_mae_data (list): Shared list to accumulate MAE data for each LA.
    """
    la_gpkg_dict = LOCAL_AUTHORITIES[la_name]
    # If this LA is actually a grouped region (e.g. Greater Manchester)
    if isinstance(la_gpkg_dict, dict):
        la_layer_name = la_gpkg_dict["gpkg_file"].replace(".gpkg", "")
        sub_las = la_gpkg_dict["sub_LAs"]
        multiple_las = True
        logging.info(f"Processing region: {la_name} with LAs: {sub_las}")
    else:
        la_layer_name = la_gpkg_dict.replace(".gpkg", "")
        sub_las = [la_name]
        multiple_las = False
        logging.info(f"Starting processing for single LA: {la_name}")

    # 1. Load the GPKG file
    la_with_desnz_hn_lsoa, la_list_of_desnz_hn_lsoas = load_transform_hn_geodata(
        desnz_hn_gpkg_path=gpkg_path,
        lsoa_shp_path=lsoa_shp_path,
        layer_name=la_layer_name,
    )
    logging.info(f"Loaded {la_name} with DESNZ heat network LSOA data.")

    # 2. Save & upload GPKG file
    save_gdf_to_gpkg(
        gdf=la_with_desnz_hn_lsoa,
        output_dir=output_dir,
        filename_prefix=la_name,
        save_to_s3=save_to_s3,
        s3_bucket=s3_bucket,
        s3_key_dir=s3_key_dir,
        subfolder="gpkg",
    )
    # 3. Process each LA individually
    for sub_la in sub_las:
        logging.info(f"Processing LA: {sub_la}")
        process_single_LA(
            la_name=sub_la,
            la_with_desnz_hn_lsoa=la_with_desnz_hn_lsoa,
            la_list_of_desnz_hn_lsoas=la_list_of_desnz_hn_lsoas,
            multiple_las=multiple_las,
            nesta_parquet_path=nesta_parquet_path,
            output_dir=output_dir,
            optional_threshold=optional_threshold,
            save_to_s3=save_to_s3,
            s3_bucket=s3_bucket,
            s3_key_dir=s3_key_dir,
            la_mae_data=la_mae_data,
        )


def process_single_LA(
    la_name: str,
    la_with_desnz_hn_lsoa: gpd.GeoDataFrame,
    la_list_of_desnz_hn_lsoas: List[str],
    multiple_las: bool,
    nesta_parquet_path: str,
    output_dir: str,
    optional_threshold: float,
    save_to_s3: bool,
    s3_bucket: str,
    s3_key_dir: str,
    la_mae_data: List[Dict[str, float]],
):
    """
    Processes a single Local Authority (LA) or sub-LA (if part of a region like Greater Manchester)
    by analysing the intersection of DESNZ heat network zones and Nesta’s heat pump suitability scores.

    **Steps:**
    1. Filters Nesta's heat pump suitability data for the LA.
    2. Computes average scores for LSOAs inside and outside DESNZ zones.
    3. Calculates Mean Absolute Error (MAE) between DESNZ and Nesta scores.
    4. Saves outputs locally and optionally uploads to S3.

    Args:
        la_name (str): Name of the LA (or sub-LA if `multiple_las=True`).
        la_with_desnz_hn_lsoa (gpd.GeoDataFrame): DESNZ heat network zones joined with LSOAs.
        la_list_of_desnz_hn_lsoas (List[str]): LSOA codes covered by DESNZ zones.
        multiple_las (bool): If True, processes a sub-LA within a region.
        nesta_parquet_path (str): Path to Nesta's suitability scores (Parquet).
        output_dir (str): Directory for saving results.
        optional_threshold (float): Minimum DESNZ coverage for inclusion (0-1).
        save_to_s3 (bool): Whether to upload outputs to S3.
        s3_bucket (str): S3 bucket name.
        s3_key_dir (str): S3 storage directory prefix.
        la_mae_data (List[Dict[str, float]]): Stores MAE and statistical results.
    """
    # 1. Filter Nesta HP suitability scores for this LA
    la_hp_suitability_scores, la_hp_suitability_lsoas = filter_la_nesta_hp_scores(
        nesta_hp_suitability_scores=nesta_parquet_path,
        local_authority=la_name,
    )
    logging.info(f"Processed Nesta HP suitability scores for {la_name}.")

    # 2. Write out the LSOA list to a JSON file
    lsoas_json_filename = (
        f"{la_name.lower().replace(' ', '_')}_hp_suitability_lsoas.json"
    )
    lsoas_json_local_file_path = os.path.join(output_dir, lsoas_json_filename)
    with open(lsoas_json_local_file_path, "w") as file:
        json.dump(la_hp_suitability_lsoas, file)

    optionally_upload_file_to_s3(
        local_file_path=lsoas_json_local_file_path,
        s3_bucket=s3_bucket,
        s3_key_dir=s3_key_dir,
        save_to_s3=save_to_s3,
        filename=lsoas_json_filename,
        subfolder="hp_suitability_lsoas",
    )

    # 3. Check LSOAs not in HP suitability scores (only if single LA)
    if not multiple_las:
        not_in_hp_suitability = set(la_list_of_desnz_hn_lsoas) - set(
            la_hp_suitability_lsoas
        )
        logging.info(
            f"[{la_name}] Number of LSOAs in DESNZ but not in HP suitability data: {len(not_in_hp_suitability)}"
        )

    # 4. Calculate and log average Nesta heat network score
    avg_hn_score = la_hp_suitability_scores["HN_N_avg_score_weighted"].mean()
    logging.info(f"[{la_name}] Average HN_N_avg_score_weighted: {avg_hn_score}")

    # 5. Add DESNZ pilot fraction and calculate averages
    (
        la_hp_scores_with_desnz,
        avg_hn_score_pilot_nonzero,
        avg_hn_score_pilot_zero,
    ) = add_DESNZ_pilot_fraction(
        la_hp_suitability_scores=la_hp_suitability_scores,
        joined_gdf=la_with_desnz_hn_lsoa,
        optional_threshold=optional_threshold,
    )
    logging.info(
        f"[{la_name}] Avg HN_N_avg_score_weighted for DESNZ_pilot_fraction > {optional_threshold}: {avg_hn_score_pilot_nonzero}"
    )
    logging.info(
        f"[{la_name}] Avg HN_N_avg_score_weighted for DESNZ_pilot_fraction = 0: {avg_hn_score_pilot_zero}"
    )

    # 6. Calculate average scores for each threshold
    thresholds = THRESHOLDS
    average_scores_df = calculate_average_scores_for_thresholds(
        la_hp_suitability_scores=la_hp_scores_with_desnz,
        thresholds=thresholds,
    )
    avg_score_parquet_filename = (
        f"{la_name.lower().replace(' ', '_')}_average_scores_by_threshold.parquet"
    )
    avg_score_parquet_filepath = os.path.join(output_dir, avg_score_parquet_filename)
    average_scores_df.write_parquet(avg_score_parquet_filepath)
    optionally_upload_file_to_s3(
        local_file_path=avg_score_parquet_filepath,
        s3_bucket=s3_bucket,
        s3_key_dir=s3_key_dir,
        save_to_s3=save_to_s3,
        filename=avg_score_parquet_filename,
        subfolder="avg_scores",
    )

    # 7. Calculate and log the Mean Absolute Error (MAE)
    la_hp_scores_with_desnz, mae_all = calculate_mae_for_all(
        hp_suitability_scores=la_hp_scores_with_desnz,
        desnz_col="DESNZ_pilot_fraction",
        nesta_hn_score_col="HN_N_avg_score_weighted",
    )
    mae_pilot_non_zero = calculate_mae_for_pilot_score(
        hp_suitability_scores_with_desnz=la_hp_scores_with_desnz,
        hn_zones=True,
    )
    mae_pilot_zero = calculate_mae_for_pilot_score(
        hp_suitability_scores_with_desnz=la_hp_scores_with_desnz,
        hn_zones=False,
    )
    logging.info(f"[{la_name}] Mean Absolute Error (MAE) for all LSOAs: {mae_all}")
    logging.info(
        f"[{la_name}] Mean Absolute Error (MAE) for DESNZ_pilot_fraction > 0: {mae_pilot_non_zero}"
    )
    logging.info(
        f"[{la_name}] Mean Absolute Error (MAE) for DESNZ_pilot_fraction = 0: {mae_pilot_zero}"
    )

    # 8. Convert sets to strings (or None if multiple_las == True)
    missing_lsoas_info = {}
    missing_lsoas_info = {
        "not_in_hp_suitability": None if multiple_las else len(not_in_hp_suitability),
        "la_list_of_desnz_hn_lsoas": (
            None if multiple_las else len(set(la_list_of_desnz_hn_lsoas))
        ),
        "la_hp_suitability_lsoas": (
            None if multiple_las else len(set(la_hp_suitability_lsoas))
        ),
    }
    # if not multiple_las:
    #    not_in_hp_suitability_str = ",".join(sorted(not_in_hp_suitability))
    #    la_list_of_desnz_hn_lsoas_str = ",".join(sorted(len(la_list_of_desnz_hn_lsoas)))
    #    la_hp_suitability_lsoas_str = ",".join(sorted(len(la_hp_suitability_lsoas)))
    # else:
    #    not_in_hp_suitability_str = None
    #    la_list_of_desnz_hn_lsoas_str = None
    #    la_hp_suitability_lsoas_str = None

    # 9. Append all metrics—including the stringified sets—to la_mae_data
    la_mae_data.append(
        {
            "Local Authority": la_name,
            "mae_all": mae_all,
            "mae_pilot_non_zero": mae_pilot_non_zero,
            "mae_pilot_zero": mae_pilot_zero,
            "avg_hn_score": avg_hn_score,
            "avg_hn_score_pilot_nonzero": avg_hn_score_pilot_nonzero,
            "avg_hn_score_pilot_zero": avg_hn_score_pilot_zero,
            **missing_lsoas_info,
        }
    )

    # 10. Save final data (Parquet + CSV)
    mae_parquet_filename = (
        f"{la_name.lower().replace(' ', '_')}_hp_suitability_scores_with_desnz.parquet"
    )
    mae_parquet_local_file_path = os.path.join(output_dir, mae_parquet_filename)
    la_hp_scores_with_desnz.write_parquet(mae_parquet_local_file_path)
    optionally_upload_file_to_s3(
        local_file_path=mae_parquet_local_file_path,
        s3_bucket=s3_bucket,
        s3_key_dir=s3_key_dir,
        save_to_s3=save_to_s3,
        filename=mae_parquet_filename,
        subfolder="hp_suitability_scores_with_desnz",
    )
    csv_output_filename = (
        f"{la_name.lower().replace(' ', '_')}_hp_suitability_scores_with_desnz.csv"
    )
    la_hp_scores_with_desnz.write_csv(os.path.join(output_dir, csv_output_filename))

    logging.info(f"[{la_name}] Finished processing. Results in {output_dir}")


# ---------------------------------------------------------------------------
# 2. Code execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process heat network zones and calculate scores."
    )
    parser.add_argument(
        "--optional_threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for DESNZ pilot fraction. Range: 0-1.",
    )
    parser.add_argument(
        "--read_in_s3",
        action="store_true",
        help="Read input files from S3.",
    )
    parser.add_argument(
        "--save_to_s3",
        action="store_true",
        help="Save output files to S3.",
    )
    args = parser.parse_args()

    # 1. Parse command-line args
    optional_threshold = args.optional_threshold
    save_to_s3 = args.save_to_s3
    read_in_s3 = args.read_in_s3

    # 2. Set up logging and output directory
    output_dir = OUTPUT_DIR
    setup_logging_and_file_path(output_dir=output_dir)

    # 3. Shared paths (LSOA shapefile, Nesta parquet)
    paths = setup_paths(read_in_s3=read_in_s3)
    lsoa_shp_path = paths["LSOA_SHP_PATH"]
    nesta_parquet_path = paths["NESTA_HP_SUITABILITY_PARQUET_PATH"]

    # 4. Create a list to store the MAE data for each LA
    la_mae_data = []

    # 5. Loop over each Local Authority in config.py
    for la_name, gpkg_filename in LOCAL_AUTHORITIES.items():
        # If this LA references a dict, use the "gpkg_file" key
        if isinstance(gpkg_filename, dict):
            gpkg_filename = gpkg_filename["gpkg_file"]
        # Build the GPKG path depending on S3 or local
        if read_in_s3:
            gpkg_path = f"s3://asf-heat-pump-suitability/heat_network_desnz_data/{gpkg_filename}"
        else:
            gpkg_path = os.path.join(
                PROJECT_DIR,
                "asf_heat_pump_suitability/analysis/hn_zones/input_data/desnz_heat_network_zone_maps/",
                gpkg_filename,
            )

        # Process the LA or region
        process_LA(
            la_name=la_name,
            gpkg_path=gpkg_path,
            lsoa_shp_path=lsoa_shp_path,
            nesta_parquet_path=nesta_parquet_path,
            output_dir=output_dir,
            optional_threshold=optional_threshold,
            save_to_s3=save_to_s3,
            s3_bucket=S3_BUCKET,
            s3_key_dir=S3_KEY_DIR,
            la_mae_data=la_mae_data,
        )

    # 6. After all LAs are processed, save the combined MAE data as CSV
    mae_df = pl.DataFrame(la_mae_data)
    la_mae_filename = "la_mae_data.csv"
    la_mae_csv_path = os.path.join(output_dir, la_mae_filename)
    mae_df.write_csv(la_mae_csv_path)
    optionally_upload_file_to_s3(
        local_file_path=la_mae_csv_path,
        s3_bucket=S3_BUCKET,
        s3_key_dir=S3_KEY_DIR,
        save_to_s3=save_to_s3,
        filename=la_mae_filename,
        subfolder="la_mae",
    )
    logging.info(f"Saved MAE data to {la_mae_csv_path}")
    logging.info("All local authorities processed.")
