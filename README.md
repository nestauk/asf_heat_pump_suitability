# asf_heat_pump_suitability

Calculate heat pump suitability scores for lower-layer super output areas (LSOAs) in England and Wales using domestic
EPC data and supplementary sources. Scores are weight-adjusted for LSOAs where possible to reduce bias.
Intermediate outputs include:

- EPC data weighted according to LSOA using Iterative Proportional Fitting to reduce bias
- EPC data enhanced with new features including: lat/lon; listed building status; building conservation zone status;
  off gas status; average garden size per MSOA; property density per LSOA
- Individual garden size estimates for UPRNs in EPC

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

## Heat pump suitability scores

One of the challenges in assessing heat pump suitability is to set criteria for what makes a home suitable for a
particular technology. We have used two sets of criteria in this project: one a “conventional” view, which we think
reflects common consensus; and one a Nesta view, which draws on our latest research. We did this for four different
technologies; air source heat pumps (ASHPs), ground source heat pumps (GSHPs), heat networks (HNs) and shared ground
loops (SGLs).

This pipeline therefore computes a conventional score and a Nesta score for each of the four tech types listed: eight
heat pump suitability scores are calculated in total per LSOA. Scores are first computed per property based on presence/
absence of certain characteristics of the property/area using a simple additive model (see table below). Scores are then
averaged per property before finally aggregating to LSOA level.

|                                                                                   | ASHP (S) | ASHP (N) | GSHP (S) | GSHP (N) | SGL (S) | SGL (N) | HN (S) | HN (N) |
| --------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | ------- | ------- | ------ | ------ |
| Is the property NOT listed?                                                       | 0.25     | 0.25     | 0.25     | 0.25     | 0.25    | 0.25    | 0.25   | 0.25   |
| Is the property NOT in a building conservation zone?                              | 0.25     | 0.25     | 0.25     | 0.25     | 0.25    | 0.25    | 0.25   | 0.25   |
| Is the EPC rating >= C?                                                           | 1        | 0        | 1        | 0        | 1       | 0       | 0      | 0      |
| Is the property NOT a flat?                                                       | 1        | 1        | 1        | 1        | 0       | 0       | 0      | 0      |
| Is the garden >10m2?                                                              | 1        | 0        | 1        | 0        | 1       | 0       | 0      | 0      |
| Is there >2m2 of external space?                                                  | 0        | 2        | 0        | 1        | 0       | 0       | 0      | 0      |
| Is it off-gas?                                                                    | 0.5      | 0.5      | 0.5      | 0.5      | 0.5     | 0.5     | 0.5    | 0.5    |
| Is this property part of a building with multiple other properties (e.g. a flat)? | 0        | 0        | 0        | 0        | 2       | 2       | 2      | 2      |
| Is there a high property density (>60 households per km2) in this LSOA?           | 0        | 0        | 0        | 0        | 2       | 2       | 0      | 0      |
| Is this property in an urban LSOA/high heat demand density LSOA?                  | 0        | 0        | 0        | 0        | 0       | 0       | 2      | 2      |
| Maximum points per property                                                       | 4        | 4        | 4        | 3        | 7       | 5       | 5      | 5      |

## Data sources and acknowledgements

This work uses [Facebook Research's balance package](https://github.com/facebookresearch/balance) and [ipfn](https://github.com/Dirguis/ipfn) to conduct iterative proportional fitting.
A comprehensive table of citations for data used in this analysis can be found in [asf_heat_pump_suitability/config/README.md](https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/config#readme).

Sarig, T., Galili, T., & Eilat, R. (2023). balance – a Python package for balancing biased data samples. https://arxiv.org/abs/2307.06024

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
