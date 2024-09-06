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
│    Scripts to weight EPC and add new features
├───sampling/
│    Scripts to generate samples of EPC data, e.g. for use in testing
├───suitability/
│    Scripts to calculate heat pump suitability from enhanced EPC data
```

## Run full pipeline to generate heat pump suitability scores

To weight the EPC data, add the new features, and calculate heat pump suitability per property / LSOA, run the following
files in order as shown below. For each script below, we list the specific inputs and outputs for the 2023 Q4 EPC
dataset as examples, but arguments can be adjusted as required:

1. `run_scripts/run_compute_epc_weights.py`

   **Purpose**: Add LSOA & MSOA data to each EPC row and weight each EPC property according to LSOA with Iterative Proportional Fitting.

   **Run**:

   `python asf_heat_pump_suitability/pipeline/run_scripts/run_compute_epc_weights.py --epc_path s3://asf-daps/lakehouse/processed/epc/deduplicated/processed_dedupl-0.parquet -y 2023 -q 4`

   **Inputs**: preprocessed deduplicated EPC dataset

   **Outputs**:

   - EPC dataset with weights: `s3://asf-heat-pump-suitability/outputs/2023Q4/20240824_2023_Q4_EPC_weighted.parquet`
   - Processing time and number of rows lost per LSOA: `s3://asf-heat-pump-suitability/outputs/2023Q4/20240824_2023_Q4_EPC_weighted_stats.parquet`

2. `run_scripts/run_add_features.py`

   **Purpose**: Add new features to the EPC dataset:

   - mean average garden size per MSOA
   - lat/lon per UPRN
   - property density per LSOA
   - off gas properties by postcode
   - listed building status per UPRN
   - England and Wales building conservation area flag per UPRN

   **Run**:

   `python asf_heat_pump_suitability/pipeline/run_scripts/run_add_features.py --epc_path s3://asf-heat-pump-suitability/outputs/2023Q4/20240824_2023_Q4_EPC_weighted.parquet -y 2023 -q 4`

   **Inputs**: EPC dataset with weights

   **Outputs**: EPC dataset with weights and features: `s3://asf-heat-pump-suitability/outputs/2023Q4/20240827_2023_Q4_EPC_weighted_features.parquet`

3. `run_scripts/run_calculate_garden_size.py`

   **Purpose**: Calculate estimated garden size for EPC UPRNs where available from INSPIRE land registry data and Microsoft building
   footprints.

   **Run**:

   `python asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_garden_size.py --epc_path s3://asf-heat-pump-suitability/outputs/2023Q4/20240827_2023_Q4_EPC_weighted_features.parquet -y 2023 -q 4 --use_mapping s3://asf-heat-pump-suitability/source_data/2023_land_parcels_with_file_polygons.geojson`

   **Inputs**: EPC dataset with weights and features

   **Outputs**: estimated garden size for EPC UPRNs. NB: output contains only UPRNs matched to a garden.
   `s3://asf-heat-pump-suitability/outputs/2023Q4/20240901_2023_Q4_EPC_garden_size_estimates_[01/02].parquet`

4. `run_scripts/run_scripts/run_process_garden_size.py`

   **Purpose**: Clean and process garden size estimate data and join to EPC data.

   **Run**:
   `python asf_heat_pump_suitability/pipeline/run_scripts/run_process_garden_size.py --epc_path s3://asf-heat-pump-suitability/outputs/2023Q4/20240827_2023_Q4_EPC_weighted_features.parquet --gardens_path s3://asf-heat-pump-suitability/outputs/2023Q4/20240904_2023_Q4_EPC_garden_size_estimates_complete.parquet`

   **Inputs**:

   - EPC dataset with weights and features
   - Garden size estimates for EPC UPRNs

     **Outputs**: EPC dataset with weights, features, and estimated garden size
     `s3://asf-heat-pump-suitability/outputs/2023Q4/20240904_2023_Q4_EPC_weighted_features_gardens.parquet`

5. `suitability/calculate_suitability.py`

   **Purpose**: Calculate heat pump suitability of properties and LSOAs for four tech types (air-source heat pumps, ground-source heat
   pumps, shared ground loops, and heat networks) using conventional view criteria and Nesta view criteria, so 8 suitability
   scores in total.

   **Run**:

   `python asf_heat_pump_suitability/pipeline/suitability/calculate_suitability.py --epc_path s3://asf-heat-pump-suitability/outputs/2023Q4/20240904_2023_Q4_EPC_weighted_features_gardens.parquet`

   **Inputs**: EPC dataset with weights and all features

   **Outputs**:

   - Heat pump suitability scores per EPC property: `s3://asf-heat-pump-suitability/outputs/2023Q4/20240830_2023_Q4_heat_pump_suitability_per_property.parquet`
   - Heat pump suitability scores per LSOA: `s3://asf-heat-pump-suitability/outputs/2023Q4/20240830_2023_Q4_heat_pump_suitability_per_lsoa.parquet`

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
