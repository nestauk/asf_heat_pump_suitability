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
from asf_heat_pump_suitability import config

# Suitability data per LSOA
suitablitity_per_lsoa_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250114_2023_Q4_heat_pump_suitability_per_lsoa.parquet"

# The suitability of properties
suitability_per_property_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250115_2023_Q4_heat_pump_suitability_per_property.parquet"

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


def get_tech_suitability_overall_4_type(
    ashp,
    gshp,
    sgl,
    hn,
    ashp_high_thresh,
    gshp_high_thresh,
    sgl_high_thresh,
    hn_high_thresh,
):
    if (ashp >= ashp_high_thresh) | (gshp >= gshp_high_thresh):
        ind_suit = True
    else:
        ind_suit = False
    if (sgl >= sgl_high_thresh) | (hn >= hn_high_thresh):
        shared_suit = True
    else:
        shared_suit = False

    if ind_suit & shared_suit:
        return "Both individual and shared suitable"
    elif ind_suit:
        return "Individual suitable only"
    elif shared_suit:
        return "Shared suitable only"
    else:
        return "Neither suitable"


def get_tech_suitability_overall_type(ashp, hn, ashp_high_thresh, hn_high_thresh):
    if ashp >= ashp_high_thresh:
        ind_suit = True
    else:
        ind_suit = False
    if hn >= hn_high_thresh:
        shared_suit = True
    else:
        shared_suit = False

    if ind_suit & shared_suit:
        return "Both ASHP and HN suitable"
    elif ind_suit:
        return "ASHP suitable only"
    elif shared_suit:
        return "HN suitable only"
    else:
        return "Neither ASHP or HN suitable"


# Ad hoc manually created categories for Plymouth
def get_tech_suitability_manual(ashp, hn):
    if ashp < 0.68:
        return "HN suitable"
    else:
        if (hn > 0.46) & (ashp < 0.82):
            return "Multiple technologies may be feasible"
        else:
            return "ASHP suitable"


if __name__ == "__main__":

    # Suitability scores + features per property

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

    # Suitability per LSOA
    suitability_data = pd.read_parquet(suitablitity_per_lsoa_file)

    # Filter data for a particular place
    place_name = "Plymouth"  # Word in the LSOA name
    plymouth_lsoas = suitability_data[
        suitability_data["lsoa_name"].apply(lambda x: place_name in str(x))
    ]
    plymouth_lsoas_list = list(plymouth_lsoas["lsoa"].unique())

    # Join with tenure data
    census_tenure = get_target.transform_df_target_tenure()
    plymouth_lsoas_tenure = plymouth_lsoas.merge(
        census_tenure.to_pandas(), how="left", on="lsoa"
    )

    # Get proportions of each tenure type
    tenure_types = ["owner-occupied", "rental (social)", "rental (private)"]
    plymouth_lsoas_tenure["total_census_tenure_properties"] = plymouth_lsoas_tenure[
        tenure_types
    ].sum(axis=1)
    for tenure_type in tenure_types:
        plymouth_lsoas_tenure[f"Proportion {tenure_type}"] = (
            plymouth_lsoas_tenure[tenure_type]
            / plymouth_lsoas_tenure["total_census_tenure_properties"]
        )

    # Add geospatial data
    plymouth_lsoas_tenure_gdf = load_ew_boundaries(plymouth_lsoas_tenure)

    # Join with lsoa features
    plymouth_lsoas_tenure_gdf = plymouth_lsoas_tenure_gdf.merge(
        lsoa_features, how="left", on="lsoa"
    )

    plymouth_lsoas_tenure_gdf = plymouth_lsoas_tenure_gdf.round(3).rename(
        columns=column_rename_dict
    )

    quantile_thresh = 0.6
    ashp_high_thresh = plymouth_lsoas_tenure_gdf["ASHP - Nesta"].quantile(
        quantile_thresh
    )
    gshp_high_thresh = plymouth_lsoas_tenure_gdf["GSHP - Nesta"].quantile(
        quantile_thresh
    )
    sgl_high_thresh = plymouth_lsoas_tenure_gdf["SGL - Nesta"].quantile(quantile_thresh)
    hn_high_thresh = plymouth_lsoas_tenure_gdf["HN - Nesta"].quantile(quantile_thresh)

    plymouth_lsoas_tenure_gdf[
        "Overall suitability type - shared/not"
    ] = plymouth_lsoas_tenure_gdf.apply(
        lambda x: get_tech_suitability_overall_4_type(
            x["ASHP - Nesta"],
            x["GSHP - Nesta"],
            x["SGL - Nesta"],
            x["HN - Nesta"],
            ashp_high_thresh,
            gshp_high_thresh,
            sgl_high_thresh,
            hn_high_thresh,
        ),
        axis=1,
    )

    plymouth_lsoas_tenure_gdf[
        "Overall suitability type"
    ] = plymouth_lsoas_tenure_gdf.apply(
        lambda x: get_tech_suitability_overall_type(
            x["ASHP - Nesta"], x["HN - Nesta"], ashp_high_thresh, hn_high_thresh
        ),
        axis=1,
    )

    plymouth_lsoas_tenure_gdf[
        "Manual suitability type"
    ] = plymouth_lsoas_tenure_gdf.apply(
        lambda x: get_tech_suitability_manual(
            x["ASHP - Nesta"],
            x["HN - Nesta"],
        ),
        axis=1,
    )

    plymouth_lsoas_tenure_gdf.to_file(
        "plymouth_lsoas_tenure_gdf_binary_suitability.geojson", driver="GeoJSON"
    )
    # smaller version without geometry
    plymouth_lsoas_tenure_gdf.drop(["geometry"], axis=1).to_csv(
        "plymouth_lsoas_tenure_gdf_binary_suitability.csv"
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
        column_rename_dict
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

    plymouth_per_prop_data_extra.write_csv("plymouth_per_prop_data_extra.csv")
