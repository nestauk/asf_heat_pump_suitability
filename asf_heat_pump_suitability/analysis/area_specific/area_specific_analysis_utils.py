import pandas as pd
import polars as pl
import geopandas as gpd

from asf_heat_pump_suitability.getters import get_target
from asf_heat_pump_suitability.getters import get_datasets

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

per_property_column_names = [
    "UPRN",
    "lsoa",
    "LATITUDE",
    "LONGITUDE",
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
]


def load_ew_boundaries(df: pd.DataFrame) -> gpd.GeoDataFrame:

    gdf = get_datasets.load_gdf_ons_lsoa_bounds()
    gdf = gdf.to_crs(epsg=4326)  # convert to WGS84

    gdf = gdf.merge(df, how="right", left_on="LSOA21CD", right_on="lsoa")

    return gdf


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


def get_census_prop_flats_for_lsoas(lsoas_list):

    # Alternative way to get proportion of flats from census
    census_prop_type = get_target.load_transform_df_target_property_type_ew()
    census_prop_type_filtered = census_prop_type.filter(
        pl.col("lsoa").is_in(lsoas_list)
    )
    census_prop_type_filtered = census_prop_type_filtered.with_columns(
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

    return census_prop_type_filtered


def enhance_features_per_lsoa(per_property_df, per_lsoa_df):
    """
    Enhance the features per LSOA by pulling in data from the per property dataset and census data.
    """

    lsoas_list = list(per_lsoa_df["lsoa"].unique())

    # Unique features per LSOA
    lsoa_features = per_property_df.drop_duplicates(subset=["lsoa"])[
        [
            "lsoa",
            "ruc_two_fold",
            "has_anchor_property",
            "heatpump_installation_percentage",
            "households_per_km2",
        ]
    ]

    # Join with lsoa features
    per_lsoa_df = per_lsoa_df.merge(lsoa_features, how="left", on="lsoa")

    flats_per_lsoa = get_lsoa_proportion_flats(per_property_df)
    per_lsoa_df = per_lsoa_df.merge(
        flats_per_lsoa[["lsoa", "%flats"]], how="left", on="lsoa"
    )

    # Alternative way to get proportion of flats from census
    census_prop_type = get_census_prop_flats_for_lsoas(lsoas_list)

    per_lsoa_df = per_lsoa_df.merge(
        census_prop_type.to_pandas()[["lsoa", "census_%flats"]],
        how="left",
        on="lsoa",
    )

    # Add tenure data
    per_lsoa_df = add_tenure_proportions(per_lsoa_df)

    return per_lsoa_df


def process_per_property_features(
    per_property_df, property_garden_size, column_rename_dict=column_rename_dict
):
    """
    Add garden size estimates, and process per property dataset
    """

    # Per property dataset (for plotting individual properties and their features)

    uprn_list = list(per_property_df["UPRN"].unique())

    property_garden_size_filtered = property_garden_size.filter(
        pl.col("UPRN").is_in(uprn_list)
    )

    per_property_df = per_property_df.join(
        property_garden_size_filtered, on="UPRN", how="left"
    )

    per_property_df = per_property_df.rename(
        {k.split("_weighted")[0]: v for k, v in column_rename_dict.items()}
    )

    per_property_df = per_property_df.with_columns(
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

    return per_property_df
