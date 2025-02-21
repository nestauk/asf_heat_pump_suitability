"""
Combine, filter and process the HP suitability data for Plymouth
"""

import pandas as pd
import polars as pl
import geojson
import s3fs

fs = s3fs.S3FileSystem()

from asf_heat_pump_suitability import config, PROJECT_DIR
from asf_heat_pump_suitability.analysis.area_specific.area_specific_analysis_utils import (
    column_rename_dict,
    load_ew_boundaries,
    per_property_column_names,
    enhance_features_per_lsoa,
    process_per_property_features,
)

import os

# Suitability data per LSOA
suitablitity_per_lsoa_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_lsoa.parquet"

# The suitability of properties
per_property_file_name = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_property.parquet"

# Garden size per property
garden_size_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/20250114_2023_Q4_EPC_garden_size_estimates_EWS_deduplicated.parquet"


# Ad hoc manually created categories for Plymouth
def get_tech_suitability_manual(ashp, hn):
    if (ashp < 0.68) & (hn > 0.7):
        return "HN suitable"
    else:
        if ashp < 0.82:
            return "Multiple technologies may be feasible"
        else:
            return "ASHP suitable"


if __name__ == "__main__":

    # Output location

    output_directory = os.path.join(PROJECT_DIR, "outputs/area_specific_analysis/")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Suitability per LSOA
    per_lsoa_df = pd.read_parquet(suitablitity_per_lsoa_file)

    # Filter data for a particular place

    place_name = "Plymouth"  # Word in the LSOA name
    plymouth_per_lsoa_df = per_lsoa_df[
        per_lsoa_df["lsoa_name"].apply(lambda x: place_name in str(x))
    ]
    plymouth_lsoas_list = list(plymouth_per_lsoa_df["lsoa"].unique())

    # Features per property

    per_property_df = pl.read_parquet(
        per_property_file_name,
        columns=per_property_column_names,
    )

    plymouth_per_property_df = per_property_df.filter(
        pl.col("lsoa").is_in(plymouth_lsoas_list)
    )

    # -----
    # Per LSOA data

    plymouth_per_lsoa_df = enhance_features_per_lsoa(
        plymouth_per_property_df.to_pandas(), plymouth_per_lsoa_df
    )

    # Get manual categorisations of HN/ASHP zones

    plymouth_per_lsoa_df = plymouth_per_lsoa_df.round(3).rename(
        columns=column_rename_dict
    )

    plymouth_per_lsoa_df["Manual suitability type"] = plymouth_per_lsoa_df.apply(
        lambda x: get_tech_suitability_manual(
            x["ASHP - Nesta"],
            x["HN - Nesta"],
        ),
        axis=1,
    )

    # Add geospatial data
    plymouth_lsoas_gdf = load_ew_boundaries(plymouth_per_lsoa_df)

    plymouth_lsoas_gdf.to_file(
        os.path.join(output_directory, "plymouth_lsoas_gdf_binary_suitability.geojson"),
        driver="GeoJSON",
    )
    # smaller version without geometry (but keeps area names)
    plymouth_lsoas_gdf.drop(["geometry"], axis=1).to_csv(
        os.path.join(output_directory, "plymouth_lsoas_gdf_binary_suitability.csv")
    )

    # -----
    # Per property dataset

    property_garden_size = pl.read_parquet(
        garden_size_file,
    )

    plymouth_per_property_df = process_per_property_features(
        plymouth_per_property_df, property_garden_size
    )

    plymouth_per_property_df.write_csv(
        os.path.join(output_directory, "plymouth_per_prop_data_extra.csv")
    )
