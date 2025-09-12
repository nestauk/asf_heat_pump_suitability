# Bespoke data and analysis

Scripts in this folder are to create datasets for bespoke analysis.

Functions in `area_specific_analysis_utils.py` can help with collating and processing the suitability data per LSOA or per property to help with the creation of these datasets.

Data created from these scripts might be used to create bespoke maps, or help with analysis of a particular area, and will be stored in `outputs/area_specific_analysis`.

## Plymouth

We created a dataset of suitability for Plymouth by running

```
python asf_heat_pump_suitability/analysis/area_specific/combine_plymouth_data.py

```

Which creates the following:

1. `plymouth_lsoas_gdf_binary_suitability.geojson` per LSOA suitability scores as well as aggregated features per LSOA (e.g. % flats).
2. `plymouth_lsoas_gdf_binary_suitability.csv`, same as above but without the LSOA geometry.
3. `plymouth_per_prop_data_extra.csv` per property suitability scores as well as estimated garden size.

## Wales

We created a dataset of suitability for Wales by running

```
python asf_heat_pump_suitability/analysis/area_specific/combine_welsh_data.py

```

Which creates the following:

1. `wales_lsoas_gdf_binary_suitability.geojson` per LSOA suitability scores as well as aggregated features per LSOA (e.g. % flats).
2. `wales_lsoas_gdf_binary_suitability.csv`, same as above but without the LSOA geometry.
3. `wales_per_prop_data_extra.csv` per property suitability scores as well as estimated garden size.

## A note on simplifying geojson files

Sometimes the geojsons created can be very large files, an online tool `https://mapshaper.org/` can help simplify these files (e.g. by 45%) without losing much noticiable definition on Flourish.
