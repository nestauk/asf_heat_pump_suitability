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

# Data transformations to enable computing feasibility
# Once column names are fixed we can add this to the pipeline directly
plymouth_data = plymouth_data.to_dummies("predicted_tenure")
plymouth_data = plymouth_data.rename(
    {
        "predicted_tenure_owner-occupied": "owner_occupied",
        "predicted_tenure_rental (social)": "social_housing",
    }
)

plymouth_data = plymouth_data.with_columns((~pl.col("use_off_gas")).alias("on_gas"))
plymouth_data = plymouth_data.with_columns(
    (~pl.col("in_listed_building")).alias("not_listed")
)
plymouth_data = plymouth_data.with_columns(
    (~pl.col("in_cons_area")).alias("not_in_conservation_area")
)

plymouth_data = plymouth_data.to_dummies("predicted_property_type")
plymouth_data = plymouth_data.rename(
    {"predicted_property_type_Flat, maisonette or apartment": "flats"}
)

plymouth_data = plymouth_data.with_columns(
    (pl.col("garden_area_m2") > 0).alias("has_outdoor_space")
)


# %%
from assign_cluster_suitability_and_feasibility import create_df_feasibility_scoring

# %%
# Dictionary of weights for computing feasibility scores for each tech type
weights = {
    "individual_ashp_feasibility": {
        "owner_occupied": 1,
        # "in_high_income_decile": 1,
        "on_gas": 1,
        "not_listed": 1,
        "not_in_conservation_area": 1,
    },
    "collective_ashp_feasibility": {
        "owner_occupied": 1,
        # "in_high_income_decile": 1,
        "on_gas": 1,
        "not_listed": 1,
        "not_in_conservation_area": 1,
        # "cluster_size": 1
    },
    "sgl_feasibility": {
        "social_housing": 1,
        "flats": 1,
        # "on_communal_heating": 1,
        "has_outdoor_space": 1,
        "not_listed": 1,
        "not_in_conservation_area": 1,
        # "cluster_size": 1
    },
    "hn_feasibility": {
        "in_hn": 1,
        # "close_to_anchor_loads": 1,
        # "close_to_city_center": 1
    },
}

# %%
# Feature columns that will help with computing feasibility
cols = [
    "owner_occupied",
    "social_housing",
    "on_gas",
    "not_listed",
    "not_in_conservation_area",
    "flats",
    "has_outdoor_space",
    "in_hn",
]

# %%
create_df_feasibility_scoring(df=plymouth_data, weights=weights, features=cols)

# %%
