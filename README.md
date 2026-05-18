# ASF Heat Pump Suitability

### Version 2.0.0 (in progress)

V2.0.0 of the `asf_heat_pump_suitability` repository generates the underlying data for Nesta's local heat planning tool aiming to:

- cluster groups of properties by technology - grouping similar neighbouring properties by the most suitable low-carbon heating technologies\*
- provide additional key information for each cluster about properties, households, and the area they are located in

\*Low-carbon heating technologies include **individual heat sources**, **networked ground source heat pumps** (also
known as **shared ground loops**), **communal heat sources** and **district heat networks**. You can read ead more about this work [here](https://www.nesta.org.uk/project-updates/a-tool-to-support-local-clean-heat-planning/)

### Version 1.0.0

V1.0.0 of the `asf_heat_pump_suitability` contains the code used to calculate heat pump suitability scores for lower-layer
super output areas (LSOAs) in England and Wales and Data Zones in Scotland using domestic EPC data and supplementary sources. Scores are
weight-adjusted for LSOAs where possible to reduce bias.

You can see [heat pump suitability scores across Great Britain in this map](https://heatpumpsuitability.dap-tools.uk/) for
air source heat pumps, ground source heat pumps, shared group loops, and heat networks. You can read more about this work [here](https://www.nesta.org.uk/project/mapping-heat-pump-suitability-across-great-britain/). Please note that following the shift in our methodology to what we've developing in v2.0.0, the underlying data and associated map are no longer being updated. The map will be decommissioned in the near future.

Source code for v1.0.0 is available under [Releases](https://github.com/nestauk/asf_heat_pump_suitability/releases).

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Clone the repo and navigate to your local repo folder
- Run `direnv allow`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`
  - Install requirements
- Run `conda activate asf_heat_pump_suitability`
- Instructions to run pipeline scripts can be found in [asf_heat_pump_suitability/pipeline/README.md](https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/pipeline#readme)

## Repository structure

```
asf_heat_pump_suitability
├───config/
│    Respository config files and global variables
│    ├─ base.yaml - data sources and global variables and constants
│    ├─ README.md - data source information, citations, and attributions
├───getters/
│    Modules with functions to load data
│    ├─ base_getters.py - generic getter functions; no specific datasets
│    ├─ load_geodata.py - load raw geodatasets with no preprocessing
│    ├─ load_boundaries.py - load census and geographical boundaries (LSOA, LA, national etc.)
│    ├─ load_data.py - load specific raw datasets using base getters
├───pipeline/
│    Subdirs with modules to process data and produce outputs
│    ├─ cluster/ - modules to group properties
│    ├─ impute/ - modules to impute missing data
│    ├─ model/ - modules to engineer features and train models
│    ├─ run/ - main scripts to run the pipeline
│    ├─ transform/ - modules to process input datasets, files named by feature (e.g. `uprns.py`)
│    ├─ README.md - instructions to run pipeline
├───research/
│    Any exploratory or analytical work; each piece of work should have its own subdir within either the `exploratory` or `analysis folders
│    ├─ analysis/ - ad-hoc analysis that uses the data from the project or is relevant to the project
│    ├─ exploratory/ - exploration of methods, datasets, or functions that may be incorporated into the data pipeline
├───utils/
│    Modules with generic utils
```

## Data sources and acknowledgements

A comprehensive table of citations for data used in this analysis can be found in [asf_heat_pump_suitability/config/README.md](https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/config#readme). See attributions below.

- Contains public sector information licensed under the [Open Government Licence v.3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- The data for this research include data provided by the Geographic Data Service (GeoDS), a Smart Data Research UK Investment: ES/Z504464/1.
- Contains Ordnance Survey data. © Crown copyright and database right 2026.
- Contains Royal Mail data © Royal Mail copyright and database right 2026.
- Contains GeoPlace data © Local Government Information House Limited copyright and database right 2026.
- Contains Historic Environment Scotland and Ordnance Survey data © Historic Environment Scotland - Scottish Charity No. SC045925 © Crown copyright and database right 2026
- Contains public sector information from Plymouth City Council licensed under the [Open Government Licence v.3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- This information is subject to Crown copyright and database rights 2026 and is reproduced with the permission of HM Land Registry. See INSPIRE index polygons [conditions of use](https://use-land-property-data.service.gov.uk/datasets/inspire#conditions).
- This work uses HM Land Registry's INSPIRE Index Polygons service. This information is subject to Crown copyright and database rights 2026 and is reproduced with the permission of HM Land Registry. The polygons (including the associated geometry, namely x, y co-ordinates) are subject to Crown copyright and database rights 2026 Ordnance Survey AC0000851063.
- This work uses designated Historic Asset GIS Data, The Welsh Historic Environment Service (Cadw), 2026, licensed under the [Open Government Licence v.3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- This work uses Historic England data © Historic England 2026. Contains Ordnance Survey data © Crown copyright and database right 2026. The Historic England GIS Data contained in this material was obtained on August 2024. The most publicly available up to date Historic England GIS Data can be obtained from HistoricEngland.org.uk.
- This work uses Registers of Scotland's land extent polygons. © Crown copyright. Reproduced with the permission of Registers of Scotland.

#### Code & software attributions

This work uses code modified from [fieldmaps/edge-extender](https://github.com/fieldmaps/edge-extender/) for use in Python. fieldmaps/edge-extender is published under an MIT License.

## License

This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Contributor guidelines and style guide

This guide outlines the conventions for function and file naming for the `asf_heat_pump_suitability` project.

### 1. Function naming conventions

All functions must follow the naming format `def a_b_cde()`:

- **a (Action):** What the function does (e.g., `load`, `transform`, `generate`, `extend`).
- **b (Return Type):** The type of object the function returns (e.g., `df`, `gdf`, `list`, `dict`).
- **cde (Description):** Short description of the function - e.g. for getters we want to specify the source and dataset.

#### Common actions ('a')

- **load:** Load raw dataframes (from S3) with no processing.
- **transform:** Process a dataframe or data.
- **generate:** Create a new dataframe or variable.
- **extend:** Add columns to a dataframe.

#### Example function names

- `load_df_osopen_uprn()`: Load raw OSOpen UPRN dataframe.
- `filter_gdf_residential_uprns()`: Filter GeoDataFrame of UPRNs to residential UPRNs only.
- `generate_gdf_non_residential_buildings()`: Generate a new GeoDataFrame containing non-residential buildings.

### 2. Docstrings and type hinting

`.py` files must start with a docstring containing a brief description of the module.

Every function must include:

- **Type Hinting:** For both function arguments and return types.
- **Docstring:** [Google-style docstring](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) with a concise explanation of the function's purpose, and `Args` and `Returns` information (including listing types).

Getters which load specific raw datasets must include name of data publisher, geographical coverage (where applicable), and, for geospatial data, coordinate reference system.

### 3. S3 source file naming

**Important:** keep track of original source locations and descriptions in `config/README.md` using the established table format for every dataset used.
Raw source data should be saved to the `asf-heat-pump-suitability` bucket in the `/local_heat_planning/inputs/` directory (and appropriate subdir, if applicable) using the format `a_vb_c_d_e`:

- **a**: Date of origin (publication date).
- **b**: Version date (if available).
- **c**: Source name (e.g. `OSOpen`, `ONS`).
- **d**: Short descriptive name (e.g. `building_geometries`).
- **e**: Geographical coverage (e.g. `UK`, `GB`, `global`).

**Example:** `v202510_OSOpenMapLocal_building_geometries_Plymouth.shp`.

See further [technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
