"""
This script processes and visualises DESNZ heat network (HN) and Nesta heat pump suitability data
for one or more Local Authorities (LAs). It iterates over LAs defined in a configuration file
(config.hnz_config).

**Steps**:
1. **Load and Preprocess Data**:
   - Reads Parquet, JSON, and (optionally) GeoPackage files for each LA.
   - Extracts unique LSOA geometries.
   - Ensures the LSOA shapefile is in the target CRS (e.g., EPSG:27700).
   - Filters LSOA geometries to just those in the current LA.

2. **Plot LA LSOA Geometries** and save to PNG/PDF.

3. **Merge Suitability Scores with Geometries**:
   - Convert to a GeoDataFrame.

4. **Plot Overlay of DESNZ Pilot Score > 0** on LA heat network zones (if a GPKG is found).

5. **Plot Absolute Error Maps**:
   - Visualises absolute error for LSOAs inside vs. outside DESNZ pilot zones.

6. **Plot Nesta HN Scores vs. DESNZ Pilot Fraction Threshold**:
   - Loads and plots average Nesta HN scores for different area coverage thresholds.
   - Fits a linear regression line and displays R².

7. **Plot ASHP vs. HN Scatter**:
    - Merges the suitability scores onto the LA data.
    - Plots ASHP vs. HN scores for all LSOAs in the LA.

**Usage**:
    python plot_comparison_of_hn_zones_all.py [--read_from_s3]

**Example**:
    # Run the script with local files only
    python plot_comparison_of_hn_zones_all.py

    # Run the script, reading files from S3
    python plot_comparison_of_hn_zones_all.py --read_from_s3
"""

import logging
import argparse
from asf_heat_pump_suitability.getters.get_datasets import (
    load_n_hn_ashp_scores,
    load_la_data,
)
from asf_heat_pump_suitability.utils.geo_utils import (
    load_and_filter_lsoa_geometries,
    merge_hp_suitability_data_with_geometries,
)
from hnz_utils.hnz_plotting_utils import (
    plot_lsoa_geometries,
    plot_overlay,
    plot_absolute_error_map,
    plot_hn_avg_score_vs_fraction_threshold,
    plot_ashp_vs_hn_scatter,
    get_global_min_max,
)
from config.hnz_config import (
    LOCAL_AUTHORITIES,
    OUTPUT_DIR,
    OUTPUT_PLOTS_DIR,
    LSOA_SHP_PATH_S3,
    TARGET_CRS,
    NESTA_HPS_PARQUET_PATHS,
    S3_BUCKET,
    S3_KEY_DIR,
    ABSOLUTE_ERROR_THRESHOLD_ABSENT,
    ABSOLUTE_ERROR_THRESHOLD_PRESENT,
)


####################
# Configurations
####################
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
INPUT_DIR = OUTPUT_DIR  # your final data files are here


####################
# Execution
####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparisons of heat network zones for all LAs."
    )
    parser.add_argument(
        "--read_from_s3",
        action="store_true",
        help="Read the input files from S3 rather than local disk.",
    )
    args = parser.parse_args()
    read_from_s3 = args.read_from_s3

    # 1. Load the *full* HP suitability dataset once for all LAs (used for global min/max + ASHP column)
    df_hp_suitability_global = load_n_hn_ashp_scores(
        nesta_hps_parquet_path=NESTA_HPS_PARQUET_PATHS, read_from_s3=read_from_s3
    )
    # 2. Compute global min/max (ASHP vs. HN) from the full dataset
    global_x_min, global_x_max, global_y_min, global_y_max = get_global_min_max(
        df_hp_suitability_global
    )

    # Also extract just the ASHP column for merging onto each LA
    df_ashp_suitability = df_hp_suitability_global[
        ["lsoa", "ASHP_N_avg_score_weighted"]
    ]
    # (We assume local LA data already have 'HN_N_avg_score_weighted' or similar.)

    # Iterate over each Local Authority in config
    for la_name, la_value in LOCAL_AUTHORITIES.items():
        # If the LA is actually a region dict (e.g., Greater Manchester), skip or handle sub-LAs separately
        if isinstance(la_value, dict):
            logging.info(f"Skipping region '{la_name}' because it has sub-LAs.")
            continue

        logging.info(f"=== Processing {la_name} ===")

        try:
            # 1. Load data (HP Parquet, LSOA JSON, average threshold Parquet)
            hp_scores_pd, la_lsoas, avg_hn_scores_df = load_la_data(
                la_name=la_name,
                input_dir=INPUT_DIR,
                s3_bucket=S3_BUCKET,
                s3_key_dir=S3_KEY_DIR,
                read_from_s3=read_from_s3,
            )

            # 2. Preprocess LSOA geometries
            la_lsoa_gdf = load_and_filter_lsoa_geometries(
                la_lsoas=la_lsoas, lsoa_shp_path=LSOA_SHP_PATH_S3, target_crs=TARGET_CRS
            )

            # 3. Plot LSOA geometries
            plot_lsoa_geometries(
                la_lsoa_geometries_gdf=la_lsoa_gdf,
                la_name=la_name,
                output_dir=OUTPUT_PLOTS_DIR,
            )

            # 4. Merge data with geometries
            la_hp_gdf = merge_hp_suitability_data_with_geometries(
                hp_suitability_scores_pd=hp_scores_pd,
                la_lsoa_geometries_gdf=la_lsoa_gdf,
                la_name=la_name,
                target_crs=TARGET_CRS,
            )

            # 5. Overlay of DESNZ pilot fraction zones
            plot_overlay(
                la_hp_suitability_gdf=la_hp_gdf,
                la_name=la_name,
                input_dir=INPUT_DIR,
                output_dir=OUTPUT_PLOTS_DIR,
            )

            # 6. Plot absolute error maps (example thresholds)
            plot_absolute_error_map(
                la_hp_suitability_gdf=la_hp_gdf,
                la_name=la_name,
                score=ABSOLUTE_ERROR_THRESHOLD_PRESENT,
                output_dir=OUTPUT_PLOTS_DIR,
            )
            plot_absolute_error_map(
                la_hp_suitability_gdf=la_hp_gdf,
                la_name=la_name,
                score=ABSOLUTE_ERROR_THRESHOLD_ABSENT,
                output_dir=OUTPUT_PLOTS_DIR,
            )

            # 7. Plot average Nesta HN score vs. fraction coverage threshold
            plot_hn_avg_score_vs_fraction_threshold(
                average_hn_scores_coverage_df=avg_hn_scores_df,
                la_name=la_name,
                output_dir=OUTPUT_PLOTS_DIR,
            )

            # 8. Merge the global ASHP column (df_ashp_suitability) onto the LA DataFrame
            la_hp_gdf_merged = la_hp_gdf.merge(
                df_ashp_suitability,
                how="left",
                left_on="LSOA21CD",  # from LA parquet
                right_on="lsoa",  # from the global CSV
            ).drop(
                columns=["lsoa"]
            )  # we only needed it for the join

            # 9. Plot ASHP vs. HN scatter with *fixed* global axes
            plot_ashp_vs_hn_scatter(
                merged_df=la_hp_gdf_merged,
                la_name=la_name,
                global_x_min=global_x_min,
                global_x_max=global_x_max,
                global_y_min=global_y_min,
                global_y_max=global_y_max,
                output_dir=OUTPUT_PLOTS_DIR,
            )

            # Print the output directory after processing each LA
            logging.info(f"Plots for {la_name} have been saved to {OUTPUT_PLOTS_DIR}")

        except Exception as err:
            logging.error(f"Error processing {la_name}: {err}")

        logging.info(f"=== Finished {la_name} ===\n")
