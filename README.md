# ASF Heat Pump Suitability

The `asf_heat_pump_suitability` repo contains the code used to calculate heat pump suitability scores for lower-layer
super output areas (LSOAs) in England and Wales and Data Zones in Scotland using domestic EPC data and supplementary sources. Scores are
weight-adjusted for LSOAs where possible to reduce bias.
Read more about the project [here](https://www.nesta.org.uk/project/mapping-heat-pump-suitability-across-great-britain/).

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

See the general repository structure depicted below. Key files are also shown.

```
asf_heat_pump_suitability
├───analysis/
│    Scripts and notebooks for specific analyses
├───config/
│    Respository config files and global variables
│    ├─ base.yaml - data sources and mappings
│    ├─ README.md - data source information, citations, and attributions
├───getters/
│    Modules with functions to load data
│    ├─ base_getters.py - generic getter functions
│    ├─ get_datasets.py - specific getter functions to load raw datasets
│    ├─ get_dno_datasets.py - specific getter functions to load and process DNO datasets
│    ├─ get_target.py - specific getter functions to load and process target data for reweighting
├───notebooks/
│    Notebooks with prototype code for pipeline
├───pipeline/
│    Subdirs with modules to process data and produce outputs
│    ├─ evaluation/ - modules for evaluation of outputs
│    ├─ prepare_features/ - modules to prepare new features for EPC
│    ├─ reweight_epc/ - modules for reweighting EPC
│    ├─ run_scripts/ - main scripts of the pipeline
│    ├─ sampling/ - modules to sample LSOAs
│    ├─ suitability/ - modules to calculate heat pump suitability scores
│    ├─ README.md - instructions to run pipeline
├───utils/
│    Modules with generic utils
```

## Heat pump suitability scores

One of the challenges in assessing heat pump suitability is to set criteria for what makes a home suitable for a
particular technology. We have used two sets of criteria in this project: one a “conventional” view, which we think
reflects common consensus; and one a Nesta view, which draws on our latest research. We did this for four different
technologies; air source heat pumps (ASHPs), ground source heat pumps (GSHPs), heat networks (HNs) and shared ground
loops (SGLs).

This pipeline therefore computes a conventional score and a Nesta score for each of the four tech types listed: eight
heat pump suitability scores are calculated in total per LSOA. Scores are first computed per property based on presence/
absence of certain characteristics of the property/area using a simple additive model (see table below). Scores are then
averaged per property and weighted\* before finally aggregating to LSOA level. Note that a property must have at least 4
of the required features to calculate heat pump suitability to be assigned a suitability score and an LSOA must have data
for at least 15 properties to be included in the final suitability per LSOA dataset.

_\*Scores will only be weighted for an LSOA if the proportion of EPC properties in that LSOA that have a weight is above a
specified threshold - the default threshold (and the threshold we have used for our published results) is 50%. Individual
properties do not receive a weight if they are missing data required for weighting._

_If the threshold is not met for a given LSOA, suitability scores for that LSOA will be unweighted and labelled as such.
Unweighted scores may not accurately represent the suitability of an LSOA for a given heating technology as a whole and
should therefore be interpreted with caution._

|                                                                                                                             | ASHP (S) | ASHP (N) | GSHP (S) | GSHP (N) | SGL (S) | SGL (N) | HN (S) | HN (N) |
| --------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | ------- | ------- | ------ | ------ |
| Is the property NOT listed?                                                                                                 | 0.25     | 0.25     | 0.25     | 0.25     | 0.25    | 0.25    | 0.25   | 0.25   |
| Is the property NOT in a protected area\*?                                                                                  | 0.25     | 0.25     | 0.25     | 0.25     | 0.25    | 0.25    | 0.25   | 0.25   |
| Is the property's EPC rating A, B or C?                                                                                     | 1        | 0        | 1        | 0        | 1       | 0       | 0      | 0      |
| Is the property NOT a flat?                                                                                                 | 1        | 1        | 1        | 1        | 0       | 0       | 0      | 0      |
| Is the property a flat?                                                                                                     | 0        | 0        | 0        | 0        | 2       | 2       | 2      | 2      |
| Is there > `10` m2 of external space at the property?                                                                       | 1        | 0        | 1        | 0        | 1       | 0       | 0      | 0      |
| Is there > `2` m2 of external space at the property?                                                                        | 0        | 2        | 0        | 1        | 0       | 0       | 0      | 0      |
| Is the property off-gas?                                                                                                    | 0.5      | 0.5      | 0.5      | 0.5      | 0.5     | 0.5     | 0.5    | 0.5    |
| Is this property in a LSOA with a high property density? (> `60` households per km2)                                        | 0        | 0        | 0        | 0        | 2       | 2       | 0      | 0      |
| Is this property in an urban LSOA?                                                                                          | 0        | 0        | 0        | 0        | 0       | 0       | 2      | 2      |
| Is this property in a LSOA with an anchor property?                                                                         | 0        | 0        | 0        | 0        | 0       | 0       | 1      | 1      |
| What proportion of properties in this LSOA could the electricity grid support to have HPs? (`x` - which is between 0 and 1) | `x`      | 0        | `x`      | 0        | `x`     | 0       | 0      | 0      |
| Maximum points per property (`x=1` for these calculations)                                                                  | 5        | 4        | 5        | 3        | 8       | 5       | 6      | 6      |

\* A "protected area" refers to building conservation zones in England and Wales and World Heritage Sites in Scotland.

## Data sources and acknowledgements

A comprehensive table of citations for data used in this analysis can be found in [asf_heat_pump_suitability/config/README.md](https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/config#readme). See attributions below.

- Contains OS data © Crown copyright and database right 2025.
- Contains Royal Mail data © Royal Mail copyright and database right 2025.
- Contains Office for National Statistics information licensed under the Open Government Licence v.3.0.
- Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
- This information is subject to Crown copyright and database rights 2025 and is reproduced with the permission of HM Land Registry. See INSPIRE index polygons [conditions of use](https://use-land-property-data.service.gov.uk/datasets/inspire#conditions).
- This work uses HM Land Registry's INSPIRE Index Polygons service. This information is subject to Crown copyright and database rights 2024 and is reproduced with the permission of HM Land Registry. The polygons (including the associated geometry, namely x, y co-ordinates) are subject to Crown copyright and database rights 2025 Ordnance Survey 100026316.
- Microsoft GlobalMLBuildingFootprints are made available under the [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/). Any rights in individual contents of the database are licensed under the [Database Contents License](http://opendatacommons.org/licenses/dbcl/1.0/\).
- This work uses designated Historic Asset GIS Data, The Welsh Historic Environment Service (Cadw), 2025, licensed under the Open Government Licence.
- This work uses Historic England data © Historic England 2025. Contains Ordnance Survey data © Crown copyright and database right 2025. The Historic England GIS Data contained in this material was obtained on August 2024. The most publicly available up to date Historic England GIS Data can be obtained from HistoricEngland.org.uk.
- This work uses data provided by the Consumer Data Research Centre, an ESRC Data Investment.
- This work uses Registers of Scotland's land extent polygons. © Crown copyright. Reproduced with the permission of Registers of Scotland.
- This work uses data from the Scottish Census © Crown copyright. Data supplied by National Records of Scotland.
- This work uses data from Electricity North West Ltd, SP Energy Networks, SSEN Distribution, UK Power Networks. Creative Commons Attribution: https://creativecommons.org/licenses/by/4.0/
- Supported by Northern Powergrid Open Data. [License](https://northernpowergrid.opendatasoft.com/p/opendatalicence/)
- Supported by NGED Open Data. [License](https://www.nationalgrid.co.uk/open-data-licence)
- This work uses [Facebook Research's balance package](https://github.com/facebookresearch/balance) and [ipfn](https://github.com/Dirguis/ipfn) to conduct iterative proportional fitting.
  Sarig, T., Galili, T., & Eilat, R. (2023). balance – a Python package for balancing biased data samples. https://arxiv.org/abs/2307.06024

## Pipeline intermediate outputs

Intermediate outputs include:

- EPC data weighted according to LSOA using Iterative Proportional Fitting to reduce bias
- EPC data enhanced with new features including: lat/lon; listed building status; building conservation zone status;
  off gas status; average garden size per MSOA; property density per LSOA
- Individual garden size estimates for UPRNs in EPC

See detailed instructions of how to run the full pipeline in the [asf_heat_pump_suitability/pipeline/README.md](https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/pipeline#readme).

## License

This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
