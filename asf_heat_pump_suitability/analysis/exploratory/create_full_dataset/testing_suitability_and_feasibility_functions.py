# %% [markdown]
# ### A notebook for testing suitability and feasibility calculations

# %%
# Package imports
import polars as pl
import numpy as np
import geopandas as gpd

# %%
# Importing Plymouth data
plymouth_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/results/plymouth_features_selected_with_clusters.parquet"
)

# %%
# The building polygon data per cluster, and the distance to anchor loads from the centre of the cluster
cluster_polygons = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons.geojson"
).to_crs(epsg=4326)

anchor_dist_df = pl.from_pandas(
    cluster_polygons[["cluster", "distance_from_anchor_property_m"]]
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
    anchor_dist_df=anchor_dist_df,
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
    "perc_imd_decile_above_avg": [55, 10, 20, 5, 40],
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
