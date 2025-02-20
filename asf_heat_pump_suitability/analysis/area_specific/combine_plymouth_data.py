"""
Combine, filter and process the HP suitability data for Plymouth
"""

import pandas as pd
import polars as pl
import geojson
import s3fs

fs = s3fs.S3FileSystem()
import geopandas as gpd

from asf_heat_pump_suitability.getters import get_target
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability import config, PROJECT_DIR

import os

# Suitability data per LSOA
suitablitity_per_lsoa_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_lsoa.parquet"

# The suitability of properties
suitability_per_property_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_property.parquet"

# Garden size per property
garden_size_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/20250114_2023_Q4_EPC_garden_size_estimates_EWS_deduplicated.parquet"

column_rename_dict = {
    "ASHP_S_avg_score_weighted": "ASHP - Conventional",
    "ASHP_N_avg_score_weighted": "ASHP - Nesta",
    "GSHP_S_avg_score_weighted": "GSHP - Conventional",
    "GSHP_N_avg_score_weighted": "GSHP - Nesta",
    "SGL_S_avg_score_weighted": "SGL - Conventional",
    "SGL_N_avg_score_weighted": "SGL - Nesta",
    "HN_S_avg_score_weighted": "HN - Conventional",
    "HN_N_avg_score_weighted": "HN - Nesta",
}


def load_ew_boundaries(df: pd.DataFrame) -> gpd.GeoDataFrame:

    gdf = get_datasets.load_gdf_ons_lsoa_bounds()
    gdf = gdf.to_crs(epsg=4326)  # convert to WGS84

    gdf = gdf.merge(df, how="right", left_on="LSOA21CD", right_on="lsoa")

    return gdf


# Ad hoc manually created categories for Plymouth
def get_tech_suitability_manual(ashp, hn):
    if (ashp < 0.68) & (hn > 0.7):
        return "HN suitable"
    else:
        if ashp < 0.82:
            return "Multiple technologies may be feasible"
        else:
            return "ASHP suitable"


def add_tenure_proportions(lsoa_df: pd.DataFrame) -> pd.DataFrame:

    # Join with tenure data
    census_tenure = get_target.transform_df_target_tenure()
    lsoa_df_tenure = lsoa_df.merge(census_tenure.to_pandas(), how="left", on="lsoa")

    # Get proportions of each tenure type
    tenure_types = ["owner-occupied", "rental (social)", "rental (private)"]
    lsoa_df_tenure["total_census_tenure_properties"] = lsoa_df_tenure[tenure_types].sum(
        axis=1
    )
    for tenure_type in tenure_types:
        lsoa_df_tenure[f"Proportion {tenure_type}"] = (
            lsoa_df_tenure[tenure_type]
            / lsoa_df_tenure["total_census_tenure_properties"]
        )

    return lsoa_df_tenure


def get_lsoa_proportion_flats(per_property_df: pd.DataFrame) -> pd.DataFrame:

    # Proportion of flats per LSOA
    per_property_df["is_flat"] = (
        per_property_df["property_type"] == "Flat, maisonette or apartment"
    ).astype(int)
    per_property_df_grouped = per_property_df.groupby("lsoa")
    num_flats_per_lsoa = per_property_df_grouped["is_flat"].sum().reset_index()
    num_props_per_lsoa = per_property_df_grouped.size().reset_index()
    flats_per_lsoa = num_flats_per_lsoa.merge(num_props_per_lsoa, on="lsoa")
    flats_per_lsoa["%flats"] = flats_per_lsoa["is_flat"] * 100 / flats_per_lsoa[0]

    return flats_per_lsoa


if __name__ == "__main__":

    # Suitability per LSOA
    suitability_data = pd.read_parquet(suitablitity_per_lsoa_file)

    # Filter data for a particular place

    place_name = "Plymouth"  # Word in the LSOA name
    plymouth_lsoas_df = suitability_data[
        suitability_data["lsoa_name"].apply(lambda x: place_name in str(x))
    ]
    plymouth_lsoas_list = list(plymouth_lsoas_df["lsoa"].unique())

    plymouth_lsoas_df = add_tenure_proportions(plymouth_lsoas_df)

    # Add geospatial data
    plymouth_lsoas_gdf = load_ew_boundaries(plymouth_lsoas_df)

    # Features per property

    per_prop_data = pl.read_parquet(
        suitability_per_property_file,
        columns=[
            "UPRN",
            "lsoa",
            "LATITUDE",
            "LONGITUDE",
            "UPRN_duplicated",
            "CURRENT_ENERGY_RATING",
            "property_type",
            "off_gas",
            "garden_area_m2",
            "ruc_two_fold",
            "households_per_km2",
            "heatpump_installation_percentage",
            "has_anchor_property",
            "tenure",
            "in_protected_area",
            "lad_conservation_area_data_available_ew",
            "listed_building",
            "build_year",
            "ASHP_S_avg_score",
            "ASHP_N_avg_score",
            "GSHP_S_avg_score",
            "GSHP_N_avg_score",
            "SGL_S_avg_score",
            "SGL_N_avg_score",
            "HN_S_avg_score",
            "HN_N_avg_score",
        ],
    )

    per_prop_plymouth = per_prop_data.filter(
        pl.col("lsoa").is_in(plymouth_lsoas_list)
    ).to_pandas()

    # Unique features per LSOA
    lsoa_features = per_prop_plymouth.drop_duplicates(subset=["lsoa"])[
        [
            "lsoa",
            "ruc_two_fold",
            "has_anchor_property",
            "heatpump_installation_percentage",
            "households_per_km2",
        ]
    ]

    # Join with lsoa features
    plymouth_lsoas_gdf = plymouth_lsoas_gdf.merge(lsoa_features, how="left", on="lsoa")

    flats_per_lsoa = get_lsoa_proportion_flats(per_prop_plymouth)
    plymouth_lsoas_gdf = plymouth_lsoas_gdf.merge(
        flats_per_lsoa[["lsoa", "%flats"]], how="left", on="lsoa"
    )

    # Alternative way to get proportion of flats from census
    census_prop_type = get_target.load_transform_df_target_property_type_ew()
    census_prop_type_plymouth = census_prop_type.filter(
        pl.col("lsoa").is_in(plymouth_lsoas_list)
    )
    census_prop_type_plymouth = census_prop_type_plymouth.with_columns(
        sum=pl.sum_horizontal(
            "Detached",
            "Semi-detached",
            "Terraced (including end-terrace)",
            "Flat, maisonette or apartment",
            "Caravan or other mobile or temporary structure",
        )
    ).with_columns(
        (pl.col("Flat, maisonette or apartment") * 100 / pl.col("sum")).alias(
            "census_%flats"
        )
    )
    plymouth_lsoas_gdf = plymouth_lsoas_gdf.merge(
        census_prop_type_plymouth.to_pandas()[["lsoa", "census_%flats"]],
        how="left",
        on="lsoa",
    )

    # Get categorisations of HN/ASHP zones

    plymouth_lsoas_gdf = plymouth_lsoas_gdf.round(3).rename(columns=column_rename_dict)

    plymouth_lsoas_gdf["Manual suitability type"] = plymouth_lsoas_gdf.apply(
        lambda x: get_tech_suitability_manual(
            x["ASHP - Nesta"],
            x["HN - Nesta"],
        ),
        axis=1,
    )

    plymouth_lsoas_gdf.to_file(
        "plymouth_lsoas_gdf_binary_suitability.geojson", driver="GeoJSON"
    )
    # smaller version without geometry
    plymouth_lsoas_gdf.drop(["geometry"], axis=1).to_csv(
        "plymouth_lsoas_gdf_binary_suitability.csv"
    )

    # -----

    # Per property dataset (for plotting individual properties and their features)

    plymouth_suit_properties = per_prop_data.filter(
        pl.col("lsoa").is_in(plymouth_lsoas_list)
    )
    plymouth_uprn_list = list(plymouth_suit_properties["UPRN"].unique())

    property_garden_size = pl.read_parquet(
        garden_size_file,
    )
    plymouth_garden = property_garden_size.filter(
        pl.col("UPRN").is_in(plymouth_uprn_list)
    )

    plymouth_per_prop_data_extra = plymouth_suit_properties.join(
        plymouth_garden, on="UPRN", how="left"
    )

    plymouth_per_prop_data_extra = plymouth_per_prop_data_extra.rename(
        {k.split("_weighted")[0]: v for k, v in column_rename_dict.items()}
    )

    plymouth_per_prop_data_extra = plymouth_per_prop_data_extra.with_columns(
        pl.col(
            [
                "ASHP - Conventional",
                "ASHP - Nesta",
                "GSHP - Conventional",
                "GSHP - Nesta",
                "SGL - Conventional",
                "SGL - Nesta",
                "HN - Conventional",
                "HN - Nesta",
                "garden_area_m2",
            ]
        ).round(3)
    )

    output_directory = os.path.join(PROJECT_DIR, "outputs/area_specific_analysis/")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plymouth_per_prop_data_extra.write_csv(
        os.path.join(output_directory, "plymouth_per_prop_data_extra.csv")
    )
