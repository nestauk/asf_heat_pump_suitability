# %% [markdown]
# ### A notebook for testing suitability and feasibility calculations

# %%
# Package imports
import polars as pl
import numpy as np

# %%
# Importing Plymouth data
plymouth_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_uprn_with_features.parquet"
)

# %%
# Generating random cluster numbers as current plymouth data doesn't have those
random_clusters = np.random.randint(low=1, high=11, size=len(plymouth_data))
plymouth_data = plymouth_data.with_columns(pl.Series("cluster", random_clusters))

# Dropping oa21 and renaming OA21CD to oa21 for now - final dataset will use oa21
plymouth_data = plymouth_data.drop("oa21").rename({"OA21CD": "oa21"})

# Renaming in_hn to in_heat_network_zone
plymouth_data = plymouth_data.rename({"in_hn": "in_heat_network_zone"})

# Renaming in_cons_area to in_conservation_area
plymouth_data = plymouth_data.rename({"in_cons_area": "in_conservation_area"})

# Create distance_to_anchor_loads column and distance_to_city_centre columns with random values in meters, as current data doesn't have those
random_distance_to_anchor_loads = np.random.randint(
    low=100, high=5000, size=len(plymouth_data)
)
random_distance_to_city_centre = np.random.randint(
    low=100, high=5000, size=len(plymouth_data)
)
plymouth_data = plymouth_data.with_columns(
    pl.Series("distance_to_anchor_loads", random_distance_to_anchor_loads),
    pl.Series("distance_to_city_centre", random_distance_to_city_centre),
)

# Create flag in_high_income_decile with random boolean values, as current data doesn't have those
random_in_high_income_decile = np.random.choice([True, False], size=len(plymouth_data))
plymouth_data = plymouth_data.with_columns(
    pl.Series("in_high_income_decile", random_in_high_income_decile)
)

# Create flag on_communal_heating with random boolean values, as current data doesn't have those
random_on_communal_heating = np.random.choice([True, False], size=len(plymouth_data))
plymouth_data = plymouth_data.with_columns(
    pl.Series("on_communal_heating", random_on_communal_heating)
)

# create a building ID column at random for now
random_building_ids = np.random.randint(low=1, high=50000, size=len(plymouth_data))
plymouth_data = plymouth_data.with_columns(
    pl.Series("building_id", random_building_ids)
)

# %% [markdown]
# ## Feasibility calculations

# %%
from assign_cluster_suitability_and_feasibility import (
    prepare_df_for_feasibility_scoring,
)
from config import features, city_centre_oas

# %%
feasibility_scoring_data = prepare_df_for_feasibility_scoring(
    df=plymouth_data,
    features=features,
    anchor_loads_threshold=500,
    outdoor_space_threshold=30,
    city_centre_oas=city_centre_oas,
)

# %%
from assign_cluster_suitability_and_feasibility import create_df_feasibility_scoring
from config import weights, expected_tech_types


# %%
weights

# %%
create_df_feasibility_scoring(
    df=feasibility_scoring_data,
    weights=weights,
    expected_tech_types=expected_tech_types,
    features=features,
)

# %%
dummy_df = {
    "cluster_size": [
        20,
        10,
        100,
        40,
        5,
    ],  # these should be values between 0 and 100, not the original cluster sizes
    "perc_owner_occupied": [60, 5, 45, 20, 33],
    "perc_social_housing": [30, 90, 40, 70, 50],
    "perc_flats": [10, 80, 30, 60, 20],
    "perc_in_high_income_decile": [55, 10, 20, 5, 40],
    "perc_on_gas": [70, 20, 80, 10, 50],
    "perc_not_in_listed_building": [100, 100, 90, 50, 30],
    "perc_not_in_conservation_area": [100, 87, 95, 60, 20],
    "perc_close_to_anchor_loads": [90, 10, 50, 76, 30],
    "perc_close_to_city_centre": [80, 5, 60, 70, 25],
    "perc_on_communal_heating": [5, 70, 20, 50, 30],
    "perc_has_outdoor_space": [95, 20, 80, 60, 40],
    "perc_in_heat_network_zone": [85, 15, 40, 70, 25],
}

dummy_df = pl.DataFrame(dummy_df)

create_df_feasibility_scoring(df=dummy_df)

# %% [markdown]
# ## Suitability calculations

# %%
from assign_cluster_suitability_and_feasibility import (
    prepare_df_for_suitability_categorisation,
)
from config import city_centre_oas

# %%
suitability_categorisation_data = prepare_df_for_suitability_categorisation(
    df=plymouth_data, city_centre_oas=city_centre_oas, outdoor_space_threshold=30
)

# %%
from assign_cluster_suitability_and_feasibility import (
    create_df_suitability_categorisation,
)

# %%
create_df_suitability_categorisation(df=suitability_categorisation_data)

# %%
dummy_df = {
    "cluster_size": [1, 30, 50, 21, 100],
    "in_heat_network_zone": [True, True, False, False, False],
    "in_city_centre": [False, True, False, False, False],
    "in_conservation_area": [False, True, False, True, False],
    "has_outdoor_space": [True, True, False, True, True],
}

dummy_df = pl.DataFrame(dummy_df)

create_df_suitability_categorisation(df=dummy_df)

# %%
