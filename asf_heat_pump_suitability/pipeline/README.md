# asf_heat_pump_suitability pipeline README

## `asf_heat_pump_suitability/pipeline` structure

```
asf_heat_pump_suitability/pipeline
├───evaluation/
│    Modules with functions for evaluating outputs, e.g. reweighting
├───prepare_features/
│    Modules with functions for preparing new features to join to EPC data
├───reweight_epc/
│    Modules with functions to prepare and conduct reweighting with IPF
├───run_scripts/
│    All run scripts to weight EPC, add new features, and calculate suitability
├───sampling/
│    Scripts to generate samples of EPC data, e.g. for use in testing
├───suitability/
│    Modules with functions to calculate heat pump suitability
```

## Run full pipeline to generate heat pump suitability scores

To calculate heat pump suitability, you first need to produce the required inputs:
To weight the EPC data, add new features, and estimate garden size of properties in preparation for calculating suitability, you can run the
following files in any order. All scripts take the preprocessed and deduplicated EPC dataset (output from `asf-daps`) in
parquet file format as input. Ensure you set the `--year` and `--quarter` arguments to correspond to those of the EPC
dataset when running each script.See the script `.py` files for more detailed running instructions.

|             Script             | Purpose                                                                                                                          | Inputs                                                                                                                                                                                                                                  | Output filename                                                                                        | Output description                                                                                                                                        |
| :----------------------------: | :------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `run_compute_epc_weights.py`  | Weight properties with Iterative Proportional Fitting per LSOA / Data Zone to reduce bias.                                       | EPC preprocessed and deduplicated parquet file from `asf-daps`.                                                                                                                                                                         | `[DATE]_[EPC_YEAR]_[EPC_Q]_EPC_weights.parquet`; `[DATE]_[EPC_YEAR]_[EPC_Q]_EPC_weights_stats.parquet` | Weighted EPC data and weighting run stats. Unweighted rows are not retained.                                                                              |
|     `run_add_features.py`      | Add new features to the EPC dataset.                                                                                             | `[DATE]_[EPC_YEAR]_[EPC_Q]_EPC_features.parquet`                                                                                                                                                                                        | EPC preprocessed and deduplicated parquet file from `asf-daps`.                                        | Full preprocessed and deduplicated EPC dataset with all features added to each record where available.                                                    |
| `run_calculate_garden_size.py` | Calculate estimated garden size for EPC UPRNs where available from INSPIRE land registry data and Microsoft building footprints. | EPC preprocessed and deduplicated parquet file from `asf-daps` and `inspire_file_bounds_[NATION(S)].geojson`. The `geojson` file contains the geospatial boundary polygons of the INSPIRE land extent files for the specified nation.\* | `[DATE]_[EPC_YEAR]_[EPC_Q]_EPC_garden_size_estimates_[NATION(S)].parquet`                              | EPC UPRNs with estimated garden sizes for the specified nation(s) (of England & Wales; Scotland; or all). UPRNs not matched to a garden are not retained. |

To calculate heat pump suitability per property / LSOA, you can then run the following file:

|             Script             | Purpose                                                                                                                                                                                                                                                           | Inputs                                                   | Output filename                                                                                                                                                                                            | Output description                                                                                                       |
| :----------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| `run_calculate_suitability.py` | Calculate heat pump suitability of properties and LSOAs for four tech types (air-source heat pumps, ground-source heat pumps, shared ground loops, and heat networks) using conventional view criteria and Nesta view criteria, so 8 suitability scores in total. | EPC weights; EPC features; and EPC garden size estimates | `[DATE]_[EPC_YEAR]_[EPC_Q]_heat_pump_suitability_per_lsoa.parquet`; `[DATE]_[EPC_YEAR]_[EPC_Q]_heat_pump_suitability_per_lsoa.csv`; `[DATE]_[EPC_YEAR]_[EPC_Q]_heat_pump_suitability_per_property.parquet` | Heat pump suitability scores for four different tech types in Nesta and 'conventional' views per property, and per LSOA. |

\*To produce the `inspire_file_bounds_[NATION(S)].geojson` file, run the files below in the given order. See the script `.py` files
for more detailed running instructions.

1. `run_stream_inspire_files.py` - stream INSPIRE land registry files for Scotland from ROS webpage and/or INSPIRE files for England and Wales
   from government website to S3 asf-heat-pump-suitability bucket. Files are unzipped during streaming and
   saved to S3 in unzipped format.
2. `run_get_inspire_file_bounds.py` - generate bounding polygons of each INSPIRE land registry file and save to S3.

## Get sample EPC datasets

To take a sample subset of the EPC dataset to work with, run the following command in terminal after navigating to the
`asf_heat_pump_suitability` root folder:
`python asf_heat_pump_suitability/pipeline/get_samples/sample_by_area.py`

The script will produce two datasets and save them to S3 in `asf-heat-pump-suitability/outputs`:

- `epc_sample_lsoa.parquet`
- `epc_sample_msoa.parquet`

The datasets are samples of the EPC dataset. They are generated by filtering the EPC dataset to sample output areas
(OAs) (lower-layer super output areas (LSOAs) or middle-layer super output areas (MSOAs)). Sample OAs are selected from
each of England, Scotland, and Wales and represent a range of n-observations per OA.
