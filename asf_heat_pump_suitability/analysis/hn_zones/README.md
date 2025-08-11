                                                                                                                                                                                                                                                                                                           |

# README for heat network score evaluation using DESNZ heat network pilot zones

We want to evaluate the suitability scores that we have calculated using our heat pump suitability dataset. To this end, the [Department for Energy Security and Net Zero's (DESNZ) heat network zone identification pilot](https://www.gov.uk/government/collections/heat-network-zoning) developed and implemented a methodology to identify areas suitable for heat networks within 28 pilot locations across England.

We have extracted geographic data (in the form of polygons in gpkg files) from [maps](https://www.gov.uk/government/publications/heat-network-zoning-maps) released in these pilots. These inputs are required for this evaluation pipeline and the extraction scripts are _not_ included in this repository. The pilot map data allows us to evaluate how Nesta's heat network suitability scores compare to the DESNZ recommendations for each local authority in the pilot zones.

## `asf_heat_pump_suitability/analysis/hn_zones` structure

```
asf_heat_pump_suitability/analysis/hn_zones
├── comparison_of_hn_zones.py
│   Script that analyses DESNZ heat network zones and Nesta's heat pump suitability data for one or
│   more local authorities (LAs). Performs spatial joins, computes coverage fractions, and calculates
│   summary statistics.
├── plot_comparison_of_hn_zones.py
│   Loads and visualises the DESNZ heat network and Nesta heat pump suitability data from
│   comparison_of_hn_zones.py for all included LAs.
├── hnz_utils/
│   Utility modules for DESNZ comparison, plotting, and logging.
├── config/
│   Contains config files centralising configuration variables (e.g. file paths and LA mappings).
```

To understand exactly how these scripts work and what they measure, take a look at our **Methodology** section below. It provides a concise overview of the steps taken in our analysis.

## Methodology

1. **Spatial Analysis**

   - Load DESNZ heat network zone polygons and LSOA boundaries, ensuring both use a consistent projection (CRS).
   - Perform a spatial intersection to calculate how much of each LSOA lies within the DESNZ pilot zone boundaries.

2. **Nesta Data Processing**

   - Load Nesta’s heat pump suitability scores and filter them for the relevant Local Authorities (LAs).
   - Compare LSOAs that lie inside vs. outside the DESNZ pilot zones.

3. **Statistical Metrics**
   - Compute average Nesta heat network scores for LSOAs within and outside DESNZ pilot areas.
   - Calculate Mean Absolute Error (MAE) to quantify differences between the DESNZ coverage fraction and Nesta’s heat network suitability score.

This script can also handle "region" entities (like Greater Manchester) by processing each constituent LA separately but following the same steps.

---

Having seen an overview of the methodology, you can now run our analysis and visualisation pipeline. The following section explains how to execute each script and what the inputs/outputs look like.

## Run pipeline to evaluate Nesta heat network scores

To perform the evaluation, we run the script `comparison_of_hn_zones.py` with the option of the following flags `--read_in_s3` and `--save_to_s3` if you want to read or save from or to an s3 bucket (e.g. `python comparison_of_hn_zones.py --read_in_s3 --save_to_s3`). These are boolean flags so if you do not select these flags, the inputs will be read locally to your machine and the outputs will be written locally.

| Script                      | Purpose                                                                                                                                                                             | Inputs                                                                                                           | Output filenames                                                                                                                                                                                                                                                                                                    | Output description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `comparison_of_hn_zones.py` | Analyse DESNZ heat network zones and Nesta heat pump suitability data for all local authorities (LAs). Performs spatial joins, computes coverage fractions, and summary statistics. | - DESNZ heat network zone pilot zone polygons (gpkg)<br>- Nesta heat pump suitability data<br>- LA boundary data | - GeoPackage: `<la_name>_with_desnz_hn_lsoa.gpkg`<br>- JSON: `<la_name>_hp_suitability_lsoas.json`<br>- Parquet: `<la_name>_hp_suitability_scores_with_desnz.parquet`<br>- CSV: `<la_name>_hp_suitability_scores_with_desnz.csv`<br>- Combined MAE CSV for all LAs: `la_mae_data.csv`<br>- Log: `script_output.log` | For each LA, outputs:<br> - A GeoPackage with the DESNZ Heat Network zones joined with the LSOA polygons and the intersection areas for each of the LAs (`la_name_with_desnz_hn_lsoa.gpkg`).<br> - A JSON file listing LSOAs (`la_name_hp_suitability_lsoas.json`).<br> - A Parquet file containing final suitability scores with DESNZ coverage (`la_name_hp_suitability_scores_with_desnz.parquet`).<br> - A CSV of those same scores (`la_name_hp_suitability_scores_with_desnz.csv`).<br> - Also creates a combined CSV (`la_mae_data.csv`) aggregating MAE metrics across all LAs as well as LSOAs which aren't contained within the LA but in the DESNZ heat network zone.<br> - Logs all steps and statistics (`script_output.log` by default). |

The script below complements the evaluation by generating visualisations that help interpret the results. We can run `plot_comparison_of_hn_zones.py` with the option of reading inputs from an s3 bucket using `--read_from_s3` (e.g. `python plot_comparison_of_hn_zones --read_from_s3`). If you do not select this boolean flag, the files will be read in locally from your machine.

| Script                           | Purpose                                                                                                                                                                                                                                                                                                                          | Inputs                                                                                                                                                                                                                                                                               | Output filenames                                                                                                                                                                                                                                                                                                     | Output description                                                                                                                                                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plot_comparison_of_hn_zones.py` | Processes and visualises DESNZ heat network (HN) and Nesta heat pump suitability data for one or more Local Authorities (LAs). Iterates over LAs defined in the config and creates various plots, including LSOA geometry maps, overlay maps, absolute error maps, coverage thresholds, and scatter plots of ASHP vs. HN scores. | - GeoPackage: `<la_name>_with_desnz_hn_lsoa.gpkg`<br>- JSON: `<la_name>_hp_suitability_lsoas.json`<br>- Parquet: `<la_name>_hp_suitability_scores_with_desnz.parquet`<br>- HP suitability Parquet <br>- LSOA boundary data (SHP)<br>- `config.hnz_config` (LA definitions and paths) | Multiple plot files (e.g., PNG or PDF) for each LA in the configured `OUTPUT_PLOTS_DIR` directory (configured in `config/hnz_config.py`):<br> - LSOA geometry maps<br> - Overlay of DESNZ pilot zones vs. HP suitability<br> - Absolute error maps<br> - HN coverage threshold plots<br> - ASHP vs. HN scatter plots | For each LA, the script:<br>- Loads HP data (GB-level + LA-level), merges with LSOA geometries.<br>- Plots and saves various PNG/PDF files visualising LSOA coverage, overlay of pilot zones, absolute error, HN vs. ASHP scatter, etc. |

---
