# 0.

Cluster the UPRNs?

# 1.

Run the notebook:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/create_full_dataset_plymouth.py`

To fill missing data.

# 2.

Run the notebook:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/Merging_clusters_plymouth.py`

To merge building polygons and create the distance from anchor property feature.

# 3.

Run the script:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/assign_cluster_suitability_and_feasibility.py`

To calculate most suitable techs and feasibility scores.

# 4.

Filter big datasets for just Stoke:
`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/filter_for_stoke_data.py`

# 5.

Format all Stoke data for creating maps in Flourish with:
`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/stoke_maps_formatting.py`
