# Block of flats classification model

This directory contains the code that classifies whether a building is a block of flats or not. The decision tree needs this
information to assign a technology type.

We trained a random forest binary classifier model which labels building footprints as either blocks of flats or not. Blocks of flats are considered to be mainly either purpose-built blocks of flats with three or more storeys (including above shops) or large buildings converted into multiple dwellings - detailed descriptions of the various archetypes considered to be blocks of flats and non-blocks can be found in [Table 1](#table-1-manual-labels). Features are generated per building from building footprint and UPRN geodata.

## Contents

- [1. Scripts](#1-scripts)
- [2. Sampling data for manual labelling](#2-sampling-data-for-manual-labelling)
- [3. Manual labelling](#3-manual-labelling)
- [4. Model training and evaluation](#4-model-training-and-evaluation)
- [5. Model application](#5-model-application)

## 1. Scripts

The block of flats classifier model is generated and trained with the [`train_model.py`](train_model.py) script. The model is trained from a manually labelled sample, an unlabelled version of which is produced with the [`sample_for_labelling.py`](sample_for_labelling.py) script. Sampling only needs to be run in the case that you wish to generate new data for manual labelling, and training needs to be run when there is new labelled data or the modelling methodology changes. Every pipeline run then loads an existing trained model to label buildings, using `--model_release` (see [`pipeline/README.md`](../../README.md)).

| Script                                               | What it does                                                                                                                                                |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`sample_for_labelling.py`](sample_for_labelling.py) | Draws a stratified sample of buildings containing flats, assigns train/test split, and saves it as one KML file per labeller, for manual labelling.         |
| [`train_model.py`](train_model.py)                   | Trains and evaluates the classifier, using functions to preprocess the data from `labelled_data.py` and to engineer features form `feature_engineering.py`. |

## 2. Sampling data for manual labelling

The `sample_for_labelling.py`(sample_for_labelling.py) script produces a stratified sample (with padding of smaller groups) of a desired sample size. This script outputs a `KML` file which can be uploaded to Google MyMaps for manual labelling.

The labelled dataset aims to represent the diversity of buildings containing flats across Great Britain. It samples according to a set of primary strata and secondary constraints to sample against which we assumed would give us a representative sample of the true population.

Train/validation and test sets of data for the model were created up front in the sampling process in order to train the model for real-world application.

### How to run the sampling

```bash
python asf_heat_pump_suitability/pipeline/model/block_of_flats/sample_for_labelling.py \
    --labellers <name1> <name2> <name3> \
    --labeller_counts <n1> <n2> <n3> \
    --local_authorities GB \
    --release_date YYYYMMDD \
    --map_uprns_to_building --save
```

| Argument                        | Required | Description                                                                                                                                           |
| ------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--labellers`                   | Yes      | Names of the people who will label the sample.                                                                                                        |
| `--labeller_counts`             | No       | Number of buildings each labeller should label, in the same order as `--labellers`. Needed with `--save`, which saves one KML file per labeller.      |
| `--local_authorities`           | No       | Local authorities to sample from. Defaults to `GB` (all of Great Britain).                                                                            |
| `--release_date`                | No       | `YYYYMMDD` release of the domestic UPRNs (from `uprns.py`) to use. Defaults to today.                                                                 |
| `--seed`                        | No       | Random seed. Defaults to 7.                                                                                                                           |
| `--target_n`                    | No       | Target number of buildings in the sample. Defaults to 3000.                                                                                           |
| `-m`, `--map_uprns_to_building` | No       | Generate the UPRN to building ID mapping from scratch and save it. Needs to be set the first time, because the training step reads the saved mapping. |
| `--save`                        | No       | Save outputs to S3, including the sample and the KML files for labelling. Without this, nothing is saved.                                             |

Outputs are saved to `outputs/models/block_of_flats_classifier/<release_date>/` on S3 (paths are set in
[`config/base.yaml`](../../../config/base.yaml)). Outputs are:

- The unlabelled sample as a parquet file.
- One KML file per labeller containing the building footprints that labeller has been assigned. 30 of each labeller's buildings (by default) are also assigned to a second labeller so that agreement between labellers can be checked. Each building has a Google Maps link to its location and its `building_id`, number of flats and number of UPRNs in the description to assist with manual labelling.
- The UPRN to building ID mapping (only with `--map_uprns_to_building`) and the enriched buildings dataset, with and without geometries.

## 3. Manual labelling

After the sample is produced, labellers manually assign one of 24 labels to each building footprint in the sample in Google MyMaps. The labels are grouped into either 'block of flats', 'non-block', or 'excluded'. Labellers must also a confidence of either 1 (confident) or 0 (not confident) to their labels. See section [2.b.iii of the methodology documentation](https://media.nesta.org.uk/documents/full_methodology_README.pdf) for more detail.

### How to record and save labels

Each labeller opens their KML file in Google MyMaps and renames each building polygon to its two-letter label followed by the confidence digit, for example `TB1` (towerblock with confidence 1). The `building_id` in the polygon description must be kept, because it is used to link the label back to the building.

The labelled KML files are saved to the `labelled_data/` directory of the block of flats classifier directory on S3 (`labelled_dir` in [`config/base.yaml`](../../../config/base.yaml)). All `.kml` files in that directory are loaded for training.

## 4. Model training and evaluation

[`train_model.py`](train_model.py) generates the features for the labelled buildings, then trains and evaluates a random forest classifier model. The model is optimised for F1 score on the positive class (blocks of flats).

```bash
python asf_heat_pump_suitability/pipeline/model/block_of_flats/train_model.py \
    --local_authorities GB \
    --model_release YYYYMMDD --uprn_release YYYYMMDD --save
```

| Argument              | Required | Description                                                                                                                                                        |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--local_authorities` | Yes      | The local authorities the labelled sample was drawn from.                                                                                                          |
| `--model_release`     | No       | `YYYYMMDD` date of the sample data. This will also be the date given to the trained model pickle. Defaults to today.                                               |
| `--uprn_release`      | No       | `YYYYMMDD` release of the domestic UPRNs output (from `uprns.py`) to use. Defaults to today.                                                                       |
| `--save`              | No       | Save the labelled data, the model data (labelled data with features), and the trained model to S3. Without this, the model is trained and evaluated but not saved. |

The sample, enriched building footprints and UPRN to building ID mapping must already have been generated by `sample_for_labelling.py` for the same `--model_release`.

With `--save`, the following are saved to S3:

- The processed labelled data which `add_features.py` reads.
- The model data: the labelled data joined to its features.
- The trained model as a pickle (`block_of_flats_building_classifier.pkl`).

## 5. Model application

The trained model is applied to the subset of buildings containing any domestic UPRNs and at least one flat to classify them into 'block of flats' and 'non-blocks'. All buildings containing any domestic UPRNs with zero flats are assigned the label 'non-block'. This happens in [`add_features.py`](../../run/add_features.py) which loads the trained model using `--model_release`.
