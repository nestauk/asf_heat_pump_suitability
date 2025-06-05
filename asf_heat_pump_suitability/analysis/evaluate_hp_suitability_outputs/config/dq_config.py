"""
Configuration file for data quality checks and analysis.

Defines:
- File paths, logging setup, and timestamped output directories.
- Expected dataset structure, including column names and types.
- Validation rules for numeric, proportion, boolean, and categorical columns.
- Thresholds for outlier detection (z-scores) and non-negative constraints.
"""

from asf_heat_pump_suitability import PROJECT_DIR
import os
from datetime import datetime
from urllib.parse import urlparse

# Path to data file
# timestamp for tracking
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# S3 URI for the data file
DATA_S3_URI = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250319_2023_Q4_heat_pump_suitability_per_lsoa.csv"

# parse bucket and key from the URI (if you still need them elsewhere)
parsed_uri = urlparse(DATA_S3_URI)
S3_BUCKET = parsed_uri.netloc
S3_KEY = parsed_uri.path.lstrip("/")

# output directory for logs (local)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "asf_heat_pump_suitability/analysis/evaluate_hp_suitability_outputs/logs",
    TIMESTAMP,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# path to use in the rest of the pipeline (read straight from S3)
DATA_PATH = DATA_S3_URI

# append timestamp to log filename
DATA_FILE_NAME = os.path.basename(S3_KEY)
LOG_FILENAME = f"{os.path.splitext(DATA_FILE_NAME)[0]}_{TIMESTAMP}_dq.log"

# Expected columns
EXPECTED_COLUMNS = [
    "lsoa",
    "ASHP_S_avg_score_weighted",
    "ASHP_N_avg_score_weighted",
    "GSHP_S_avg_score_weighted",
    "GSHP_N_avg_score_weighted",
    "SGL_S_avg_score_weighted",
    "SGL_N_avg_score_weighted",
    "HN_S_avg_score_weighted",
    "HN_N_avg_score_weighted",
    "scores_weighted",
    "n_properties",
    "property_density_km2",
    "rural_urban_class",
    "has_anchor_property",
    "heatpump_installation_percentage",
    "median_garden_estimate_m2",
    "proportion_in_conservation_area",
    "proportion_listed_building",
    "proportion_epc_c_plus",
    "proportion_off_gas",
    "census_proportion_flats",
    "lsoa_name",
]

# Numeric columns
NUMERIC_COLUMNS = [
    "ASHP_S_avg_score_weighted",
    "ASHP_N_avg_score_weighted",
    "GSHP_S_avg_score_weighted",
    "GSHP_N_avg_score_weighted",
    "SGL_S_avg_score_weighted",
    "SGL_N_avg_score_weighted",
    "HN_S_avg_score_weighted",
    "HN_N_avg_score_weighted",
    "n_properties",
    "property_density_km2",
    "heatpump_installation_percentage",
    "median_garden_estimate_m2",
    "proportion_in_conservation_area",
    "proportion_listed_building",
    "proportion_epc_c_plus",
    "proportion_off_gas",
    "census_proportion_flats",
]


# Proportion columns expected to be between [0, 1]
PROPORTION_COLUMNS = [
    "proportion_in_conservation_area",
    "proportion_listed_building",
    "proportion_epc_c_plus",
    "proportion_off_gas",
    "census_proportion_flats",
]

# Boolean columns
BOOLEAN_COLUMNS = [
    "scores_weighted",
    "has_anchor_property",
]

# Categorical columns and allowed values
CATEGORICAL_COLUMNS = {
    "rural_urban_class": ["Rural", "Urban"],
}

# Outlier detection threshold (z-score)
OUTLIER_ZSCORE_THRESHOLD = 3.0

# Score columns expected to be in [0,1]
SCORE_COLUMNS = [
    "ASHP_S_avg_score_weighted",
    "ASHP_N_avg_score_weighted",
    "GSHP_S_avg_score_weighted",
    "GSHP_N_avg_score_weighted",
    "SGL_S_avg_score_weighted",
    "SGL_N_avg_score_weighted",
    "HN_S_avg_score_weighted",
    "HN_N_avg_score_weighted",
]
