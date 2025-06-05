"""
Configuration file for data quality checks and analysis.

Defines:
- S3 URI for the dataset.
- Expected dataset structure, including column names and types.
- Validation rules for numeric, proportion, boolean, and categorical columns.
- Thresholds for outlier detection (z-scores) and non-negative constraints.
"""

# S3 URI for the data file
DATA_S3_URI = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250319_2023_Q4_heat_pump_suitability_per_lsoa.csv"

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
