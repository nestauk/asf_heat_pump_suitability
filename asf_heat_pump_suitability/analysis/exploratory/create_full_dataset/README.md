## Plymouth data

To get the data together into a state for showing Flourish visualisations during a workshop on the 8/9th September 2025 the following pipeline was followed.

First, convert any of the notebook scipts to `ipynb` files using:

```
pip install jupytext
jupytext --to notebook asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/create_full_dataset_plymouth.py

```

### Step 0

Cluster the UPRNs?

### Step 1

Run the notebook:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/create_full_dataset_plymouth.py`

to fill missing data.

### Step 2

Run the notebook:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/Merging_clusters_plymouth.py`

To merge building polygons and create the distance from anchor property feature.

### Step 3

Run the script:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/assign_cluster_suitability_and_feasibility.py`

To calculate most suitable technologies and feasibility scores.

### Step 4

Filter big datasets (green space and anchor loads) for just the Stoke ward of Plymouth by running the script:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/filter_for_stoke_data.py`

### Step 5

Run the notebook:

`asf_heat_pump_suitability/analysis/exploratory/create_full_dataset/stoke_maps_formatting.py`

To format the Stoke data for creating two maps in Flourish.

#### 1. Which technologies are most suitable for each area?

This map used two datasets, one for the regions and one for points on the map:

Regions (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_plot_data_all_region_types.geojson`):

- Clusters, suitability and feasibility features and scores
- Green space
- Boundary for stoke
- HN zones
- Just most suitable tech data

Points (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_anchorloads.geojson`)

- Anchor properties (not created here)

#### 2. Where should I act first?

This map used two datasets, one for the regions and one for points on the map:

Regions (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_merged_data.geojson`)

- Feasibility scores and features per cluster

Points (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_uprn_data.csv`)

- UPRNs with features and a bit of jitter added to lat/long
