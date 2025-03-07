"""
Create the datasets needed for the Flourish plots for the data story.

This involves loading and aggregating the per property and per LSOA datasets.

python -i asf_heat_pump_suitability/analysis/data_story/create_flourish_data.py

"""

import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from asf_heat_pump_suitability.getters import get_target, get_datasets
from asf_heat_pump_suitability import PROJECT_DIR

import os

# The core columns we use
per_lsoa_suit_rename_dict = {
    "ASHP_S_avg_score_weighted": "ASHP - Conventional",
    "ASHP_N_avg_score_weighted": "ASHP - Nesta",
    "GSHP_S_avg_score_weighted": "GSHP - Conventional",
    "GSHP_N_avg_score_weighted": "GSHP - Nesta",
    "SGL_S_avg_score_weighted": "SGL - Conventional",
    "SGL_N_avg_score_weighted": "SGL - Nesta",
    "HN_S_avg_score_weighted": "HN - Conventional",
    "HN_N_avg_score_weighted": "HN - Nesta",
}

suit_columns = list(per_lsoa_suit_rename_dict.values())
nesta_suit_columns = [r for r in suit_columns if "- Nesta" in r]


def get_region_data():
    """
    Get region information per LSOA, merge and standardise different
    datasets together
    """
    lad_lookup_ew = get_datasets.load_df_gov_LSOA_LA()

    region_lookup_ew = get_datasets.load_df_gov_LSOA_region()

    region_lookup_scot = get_datasets.load_df_scot_gov_data_zone_LA()

    lad_lookup_ew.rename(
        columns={
            "LSOA21CD": "lsoa",
            "LAD23NM": "LA_Name",
            "LAD23CD": "LA_Code",
        },
        inplace=True,
    )

    region_lookup_ew.rename(
        columns={"LSOA21CD": "lsoa", "RGN22NM": "region_name"}, inplace=True
    )

    region_lookup_scot.rename(
        columns={
            "DZ22_Code": "lsoa_22",
            "DZ22_Name": "lsoa_22_name_scotland",
            "SPD_Name": "sub_region_name",
        },
        inplace=True,
    )

    gdf = get_datasets.load_gdf_scotgov_data_zone_bounds(columns=["DataZone", "Name"])

    region_lookup_ew = region_lookup_ew.merge(lad_lookup_ew, on="lsoa", how="outer")

    scot_2011_lsoa = gdf[["DataZone", "Name"]].rename(
        columns={"DataZone": "lsoa", "Name": "lsoa_name_scotland"}
    )

    extra_scot_info = scot_2011_lsoa.merge(
        region_lookup_scot,
        left_on="lsoa_name_scotland",
        right_on="lsoa_22_name_scotland",
        how="left",
    )
    extra_scot_info["region_name"] = "Scotland"

    region_look_up = pd.concat([region_lookup_ew, extra_scot_info])

    return region_look_up


def get_prop_urban(gd):
    return sum(gd == "Urban") / len(gd)


def get_prop_anchor(gd):
    return sum(gd == True) / len(gd)


def get_aggregated_info_per_group(suitability_data, groupby_col):

    urban_prop_by_group = (
        suitability_data.groupby(groupby_col)["ruc_two_fold"]
        .apply(lambda x: get_prop_urban(x))
        .reset_index()
        .rename(columns={"ruc_two_fold": "prop_urban"})
    )
    anchor_prop_by_group = (
        suitability_data.groupby(groupby_col)["has_anchor_property"]
        .apply(lambda x: get_prop_anchor(x))
        .reset_index()
        .rename(columns={"has_anchor_property": "prop_with_anchor"})
    )

    group_averages = (
        suitability_data.groupby(groupby_col)[
            [
                "heatpump_installation_percentage",
                "households_per_km2",
                "census_%flats",
                "off_gas_averaged",
                "listed_building_averaged",
                "in_protected_area_averaged",
                "garden_area_m2_averaged",
                "CURRENT_ENERGY_RATING_c_above_averaged",
                "social_rental_averaged",
            ]
            + suit_columns
        ]
        .mean()
        .reset_index()
    )
    group_stds = suitability_data.groupby(groupby_col)[suit_columns].std().reset_index()
    group_n_props = suitability_data.groupby(groupby_col)["n_properties"].sum()

    group_averages = group_averages.merge(group_n_props, on=groupby_col)
    group_averages = group_averages.merge(urban_prop_by_group, on=groupby_col)
    group_averages = group_averages.merge(anchor_prop_by_group, on=groupby_col)
    group_averages = group_averages.merge(
        group_stds, on=groupby_col, suffixes=(None, "_std")
    )

    return group_averages


def groupby_epc_averages(data_per_prop):
    grouped_data = data_per_prop.group_by("CURRENT_ENERGY_RATING").agg(
        [
            pl.len().alias("num_UPRN"),
            pl.mean("ASHP_N_avg_score").alias("ASHP_N_avg_score_averaged"),
            pl.mean("HN_N_avg_score").alias("HN_N_avg_score_averaged"),
        ]
    )
    return grouped_data.to_pandas()


if __name__ == "__main__":

    output_directory = os.path.join(PROJECT_DIR, "outputs/data_story/")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # ----- Import data -----

    # Suitability data per LSOA
    suitablitity_per_lsoa_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250305_2023_Q4_heat_pump_suitability_per_lsoa.parquet"
    suitability_data = pd.read_parquet(suitablitity_per_lsoa_file)

    # Suitability data per property
    suitability_per_property_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250305_2023_Q4_heat_pump_suitability_per_property.parquet"
    per_prop_data = pl.read_parquet(
        suitability_per_property_file,
    )

    # ----- Get features per LSOA -----

    # - Some are per LSOA anyway (e.g. rural/urban)
    # - Some need to be read from census data
    # - Some need to be averaged from properties (e.g. average EPC)

    # LSOA features

    lsoa_features = per_prop_data.unique(subset=["lsoa"])

    lsoa_features = lsoa_features.to_pandas()[
        [
            "lsoa",
            "ruc_two_fold",
            "has_anchor_property",
            "heatpump_installation_percentage",
            "households_per_km2",
        ]
    ]

    # Averaged LSOA features

    per_prop_data = per_prop_data.with_columns(
        pl.col("in_protected_area").fill_null(False).alias("in_protected_area_binary"),
        pl.when(pl.col("CURRENT_ENERGY_RATING").is_in(["A", "B", "C"]))
        .then(1)
        .otherwise(0)
        .alias("CURRENT_ENERGY_RATING_c_above"),
        pl.when(pl.col("tenure") == "rental (social)")
        .then(1)
        .otherwise(0)
        .alias("social_rental"),
        pl.when(pl.col("property_type") == "Flat, maisonette or apartment")
        .then(1)
        .otherwise(0)
        .alias("flat"),
        pl.when(pl.col("ruc_two_fold") == "Urban").then(1).otherwise(0).alias("urban"),
    )

    per_lsoa_average_features = per_prop_data.group_by("lsoa").agg(
        [
            pl.mean(
                "off_gas",
            ).alias("off_gas_averaged"),
            pl.mean("listed_building").alias("listed_building_averaged"),
            pl.mean("in_protected_area_binary").alias("in_protected_area_averaged"),
            pl.mean("garden_area_m2").alias("garden_area_m2_averaged"),
            pl.mean("CURRENT_ENERGY_RATING_c_above").alias(
                "CURRENT_ENERGY_RATING_c_above_averaged"
            ),
            pl.mean("social_rental").alias("social_rental_averaged"),
        ]
    )

    lsoa_features_all = lsoa_features.merge(
        per_lsoa_average_features.to_pandas(), on="lsoa"
    )

    # LSOA features from census
    # - tenure
    # - proportion of flats

    # Get proportion of flats per LSOA
    census_prop_type = get_target.transform_df_target_property_type()

    census_prop_type = census_prop_type.with_columns(
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

    lsoa_features_all = lsoa_features_all.merge(
        census_prop_type.to_pandas()[["lsoa", "census_%flats"]],
        how="left",
        on="lsoa",
    )

    # Merge suitability and features per LSOA

    suitability_data = suitability_data.merge(lsoa_features_all, on="lsoa", how="left")

    region_look_up = get_region_data()
    suitability_data = suitability_data.merge(region_look_up, on="lsoa", how="left")
    suitability_data["lsoa_name"] = suitability_data["lsoa_name"].fillna(
        value=suitability_data["lsoa_name_scotland"]
    )

    # Create bins for the population density

    quantile_bins = [
        0,
        suitability_data["households_per_km2"].quantile(0.333),
        suitability_data["households_per_km2"].quantile(0.666),
        suitability_data["households_per_km2"].quantile(1),
    ]
    suitability_data["households_per_km2_cuts"] = pd.cut(
        x=suitability_data["households_per_km2"],
        bins=quantile_bins,
        labels=[0, 1, 2],
    )

    # Finalise and save

    suitability_data = suitability_data.rename(columns=per_lsoa_suit_rename_dict)

    suitability_data.to_csv(
        os.path.join(output_directory, "data_story_suitability_data_per_lsoa.csv"),
        index=False,
    )
    suitability_data[suitability_data["households_per_km2_cuts"].isin([1])].to_csv(
        os.path.join(
            output_directory, "data_story_suitability_data_per_lsoa_midhigh_density.csv"
        ),
        index=False,
    )

    # ----- Correlations -----

    suitability_data.corr(numeric_only=True).to_csv(
        os.path.join(output_directory, "correlations_features_per_lsoa.csv")
    )

    # ----- Averages per EPC -----

    per_epc_averages = per_prop_data.group_by("CURRENT_ENERGY_RATING").agg(
        [
            pl.mean(
                "off_gas",
            ).alias("off_gas_averaged"),
            pl.mean("listed_building").alias("listed_building_averaged"),
            pl.mean("in_protected_area_binary").alias("in_protected_area_averaged"),
            pl.mean("garden_area_m2").alias("garden_area_m2_averaged"),
            pl.mean("social_rental").alias("social_rental_averaged"),
            pl.mean("flat").alias("flat_averaged"),
            pl.mean("urban").alias("urban_averaged"),
            pl.len().alias("num_UPRN"),
            pl.mean("ASHP_S_avg_score").alias("ASHP_S_avg_score_averaged"),
            pl.mean("ASHP_N_avg_score").alias("ASHP_N_avg_score_averaged"),
            pl.mean("GSHP_S_avg_score").alias("GSHP_S_avg_score_averaged"),
            pl.mean("GSHP_N_avg_score").alias("GSHP_N_avg_score_averaged"),
            pl.mean("SGL_S_avg_score").alias("SGL_S_avg_score_averaged"),
            pl.mean("SGL_N_avg_score").alias("SGL_N_avg_score_averaged"),
            pl.mean("HN_S_avg_score").alias("HN_S_avg_score_averaged"),
            pl.mean("HN_N_avg_score").alias("HN_N_avg_score_averaged"),
        ]
    )

    per_epc_averages = per_epc_averages.to_pandas()
    per_epc_averages.sort_values(by="CURRENT_ENERGY_RATING").to_csv(
        os.path.join(output_directory, "data_story_suitability_data_per_epc.csv"),
        index=False,
    )

    # ----- Averages per region -----

    region_averages = get_aggregated_info_per_group(
        suitability_data, groupby_col="region_name"
    )

    region_averages.to_csv(
        os.path.join(output_directory, "data_story_region_averages.csv"), index=False
    )

    # Format for drop down options in plotting
    col_rename = {k + "_std": k for k in suit_columns}
    pd.melt(
        region_averages[
            ["region_name", "n_properties"] + list(col_rename.values())
        ].rename(columns=col_rename),
        id_vars=["region_name", "n_properties"],
        value_vars=col_rename.values(),
    ).to_csv(
        os.path.join(output_directory, "data_story_region_std_melt.csv"), index=False
    )

    # ----- Averages per LA -----

    la_averages = get_aggregated_info_per_group(suitability_data, groupby_col="LA_Name")

    # Add in LA code and region name
    ls_to_region_dict = dict(
        zip(suitability_data["LA_Name"], suitability_data["region_name"])
    )
    la_averages["region_name"] = la_averages["LA_Name"].map(ls_to_region_dict)

    la_name_code_dict = dict(zip(region_look_up["LA_Name"], region_look_up["LA_Code"]))
    la_averages["LA_Code"] = la_averages["LA_Name"].map(la_name_code_dict)

    # Calculate coefficient of variation (std/mean)
    for col in suit_columns:
        la_averages[f"{col}_cov"] = la_averages[f"{col}_std"] / la_averages[col]

    la_averages.rename(
        columns={
            r + "_std": f"Variation in {r.split(' -')[0]} suitability scores"
            for r in nesta_suit_columns
        }
    ).to_csv(os.path.join(output_directory, "data_story_la_averages.csv"), index=False)

    # Format for drop down options in plotting
    la_avg_melt = pd.melt(
        la_averages[["LA_Name", "region_name"] + nesta_suit_columns],
        id_vars=["LA_Name", "region_name"],
        value_vars=nesta_suit_columns,
    )
    la_avg_melt.rename(columns={"value": "Average score"}, inplace=True)
    la_avg_melt["variable"] = la_avg_melt["variable"].apply(lambda x: x.split(" -")[0])

    la_std_melt = pd.melt(
        la_averages[
            ["LA_Name", "region_name"] + [c + "_std" for c in nesta_suit_columns]
        ],
        id_vars=["LA_Name", "region_name"],
        value_vars=[c + "_std" for c in nesta_suit_columns],
    )
    la_std_melt.rename(columns={"value": "Variation in score"}, inplace=True)
    la_std_melt["variable"] = la_std_melt["variable"].apply(lambda x: x.split(" -")[0])

    la_avg_melt = la_avg_melt.merge(
        la_std_melt, on=["LA_Name", "region_name", "variable"]
    )

    la_avg_melt.to_csv(
        os.path.join(output_directory, "data_story_la_avg_std_melt.csv"), index=False
    )

    cov_cols = [r + "_cov" for r in suit_columns]

    col_rename = {k: k.split("_cov")[0] for k in cov_cols}

    la_av_melt = pd.melt(
        la_averages[["LA_Name", "region_name", "n_properties"] + cov_cols].rename(
            columns=col_rename
        ),
        id_vars=["LA_Name", "region_name", "n_properties"],
        value_vars=col_rename.values(),
    )
    la_av_melt.to_csv(
        os.path.join(output_directory, "data_story_la_std_melt.csv"), index=False
    )

    # ----- Investigate LSOA scores within a target LA - Norwich -----

    norwich_suitability = suitability_data[suitability_data["LA_Name"] == "Norwich"]

    norwich_lsoa_list = norwich_suitability["lsoa"].unique()
    norwich_per_prop = per_prop_data.filter(pl.col("lsoa").is_in(norwich_lsoa_list))

    norwich_per_prop.to_pandas().to_csv(
        os.path.join(output_directory, "norwich_per_prop.csv")
    )

    norwich_suitability["majority_social_rentals"] = (
        norwich_suitability["social_rental_averaged"] > 0.5
    )

    norwich_suitability.to_csv(
        os.path.join(output_directory, "norwich_suitability.csv")
    )

    # Add polygons for mapping
    gdf = get_datasets.load_gdf_ons_lsoa_bounds()
    gdf = gdf.to_crs(epsg=4326)  # convert to WGS84

    norwich_gdf = gdf.merge(
        norwich_suitability, how="right", left_on="LSOA21CD", right_on="lsoa"
    )
    norwich_gdf.to_file(
        os.path.join(output_directory, "norwich_gdf.geojson"), driver="GeoJSON"
    )

    # HN suit and EPC for Norwich
    # social housing vs not

    per_epc_norwich = groupby_epc_averages(norwich_per_prop)
    per_epc_norwich_social = groupby_epc_averages(
        norwich_per_prop.filter(pl.col("social_rental") == 1)
    )
    per_epc_norwich_not_social = groupby_epc_averages(
        norwich_per_prop.filter(pl.col("social_rental") == 0)
    )

    per_epc_norwich_together = per_epc_norwich.merge(
        per_epc_norwich_social,
        on="CURRENT_ENERGY_RATING",
        suffixes=(None, "_social_housing"),
    )
    per_epc_norwich_together = per_epc_norwich_together.merge(
        per_epc_norwich_not_social,
        on="CURRENT_ENERGY_RATING",
        suffixes=(None, "_not_social_housing"),
    )
    per_epc_norwich_together = per_epc_norwich_together.merge(
        per_epc_averages[
            [
                "CURRENT_ENERGY_RATING",
                "num_UPRN",
                "ASHP_N_avg_score_averaged",
                "HN_N_avg_score_averaged",
            ]
        ],
        on="CURRENT_ENERGY_RATING",
        suffixes=(None, "_all_GB"),
    )

    per_epc_norwich_together.sort_values(by="CURRENT_ENERGY_RATING").to_csv(
        os.path.join(
            output_directory, "data_story_suitability_data_per_epc_norwich.csv"
        ),
        index=False,
    )

    # Target particular LSOAs

    lsoas_interest = [
        "Norwich 011A",
        "Norwich 011B",
        "Norwich 011H",
        "Norwich 017C",
        "Norwich 009C",
    ]

    norwich_suitability_interest = suitability_data[
        suitability_data["lsoa_name"].isin(lsoas_interest)
    ]

    norwich_suitability_interest = gdf.merge(
        norwich_suitability_interest, how="right", left_on="LSOA21CD", right_on="lsoa"
    )

    norwich_suitability_interest.round(3).to_file(
        os.path.join(output_directory, "norwich_suitability_interest.geojson"),
        driver="GeoJSON",
    )

    norwich_per_prop_interest = per_prop_data.filter(
        pl.col("lsoa").is_in(norwich_suitability_interest["LSOA21CD"].unique())
    )

    norwich_per_prop_interest.to_pandas().round(3).to_csv(
        os.path.join(output_directory, "norwich_per_prop_interest.csv")
    )

    # ----- Suitability for Caerphilly and Argyll and Bute -----

    suitability_data[
        suitability_data["LA_Name"].isin(["Caerphilly", "Argyll and Bute"])
    ].to_csv(
        os.path.join(
            output_directory,
            "data_story_suitability_data_per_lsoa_caerphilly_argyll.csv",
        )
    )
