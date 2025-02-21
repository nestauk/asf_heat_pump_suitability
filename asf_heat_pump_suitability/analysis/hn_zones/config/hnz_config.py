"""
Configuration settings for Heat Network Zone (HNZ) analysis.

This module centralises configuration variables, including file paths,
default parameters, and local authority mappings. It enables easy customisation
without modifying multiple scripts.

**Key Configurations:**
- **File Paths**: Defines local and S3 paths for LSOA shapefiles and Nesta heat pump suitability data.
- **Local Authorities Mapping**: Maps Local Authority names to corresponding heat network zone GeoPackage files.
- **Data Corrections**: Adjusts inconsistencies in LA names for data alignment.
- **S3 Storage**: Configures S3 bucket names and directory paths for saving results.
- **Threshold Settings**: Stores the default DESNZ pilot fraction threshold and predefined threshold levels for score calculations.

By keeping all configuration details in one place, this module improves maintainability and adaptability for future updates.
"""

import os
from asf_heat_pump_suitability import PROJECT_DIR

# Output Directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_data/")

# LOCAL AUTHORITY DICTIONARIES
LOCAL_AUTHORITIES = {
    "Birmingham": "heat-network-zone-map-Birmingham.gpkg",
    "Bristol": "heat-network-zone-map-Bristol.gpkg",
    "Cheltenham": "heat-network-zone-map-Cheltenham.gpkg",
    "Coventry": "heat-network-zone-map-Coventry.gpkg",
    "Exeter": "heat-network-zone-map-Exeter-v2.gpkg",
    "Gateshead": "heat-network-zone-map-Gateshead.gpkg",
    "Hull": "heat-network-zone-map-Hull.gpkg",
    "Leeds": "heat-network-zone-map-Leeds.gpkg",
    "Leicester": "heat-network-zone-map-Leicester.gpkg",
    "Liverpool": "heat-network-zone-map-Liverpool.gpkg",
    "London Borough of Barking and Dagenham": "heat-network-zone-map-London-Borough-of-Barking-and-Dagenham.gpkg",
    "Newcastle upon Tyne": "heat-network-zone-map-Newcastle-upon-Tyne.gpkg",
    "Nottingham": "heat-network-zone-map-Nottingham.gpkg",
    "Peterborough": "heat-network-zone-map-Peterborough.gpkg",
    "Plymouth": "heat-network-zone-map-Plymouth.gpkg",
    "Sheffield": "heat-network-zone-map-Sheffield.gpkg",
    "Southampton": "heat-network-zone-map-Southampton.gpkg",
    "Southwark": "heat-network-zone-map-Southwark.gpkg",
    "Stoke-on-Trent": "heat-network-zone-map-Stoke-on-Trent.gpkg",
    "Sunderland": "heat-network-zone-map-Sunderland.gpkg",
    "Greater Manchester": {
        "gpkg_file": "heat-network-zone-map-Greater-Manchester.gpkg",
        "sub_LAs": [
            "Bolton",
            "Bury",
            "Manchester",
            "Oldham",
            "Rochdale",
            "Stockport",
            "Salford",
            "Tameside",
            "Trafford",
            "Wigan",
        ],
    },
}
LA_CORRECTIONS = {
    "London Borough of Barking and Dagenham": "Barking and Dagenham",
    "Hull": "Kingston upon Hull",
}

# Read in file paths for LSOA shapes as well as Nesta heat pump suitability (s3 and local option)
LSOA_SHP_PATH_LOCAL = os.path.join(
    PROJECT_DIR,
    "asf_heat_pump_suitability/analysis/hn_zones/input_data/lsoa_shape_file/LSOA_2021_EW_BFE_V9.shp",
)
LSOA_SHP_PATH_S3 = "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"

NESTA_HPS_PARQUET_LOCAL = os.path.join(
    PROJECT_DIR,
    "asf_heat_pump_suitability/analysis/hn_zones/input_data/nesta_heat_network_suitability/20240925_2023_Q4_EPC_heat_pump_suitability_per_lsoa.parquet",
)
NESTA_HPS_PARQUET_S3 = "s3://nesta-open-data/asf_heat_pump_suitability/2023Q4/20240925_2023_Q4_EPC_heat_pump_suitability_per_lsoa.parquet"

S3_BUCKET = "asf-heat-pump-suitability"
S3_KEY_DIR = "evaluation/desnz_hn_zone_scores/"

# Default DESNZ Pilot Fraction Threshold
DEFAULT_THRESHOLD = 0.0  # This can be modified here instead of in multiple scripts

# List of Thresholds for Score Calculation
THRESHOLDS = [round(i * 0.05, 2) for i in range(0, 20)]  # [0.0, 0.05, ..., 0.95]
