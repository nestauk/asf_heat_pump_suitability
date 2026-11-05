# Block of flats classification model

This directory contains the code that classifies whether a building is a block of flats or not. The decision tree needs this
information to assign a technology type.

We trained a random forest binary classifier model which labels building footprints as either blocks of flats or not. Blocks of flats are considered to be mainly either purpose-built blocks of flats with three or more storeys (including above shops) or large buildings converted into multiple dwellings - detailed descriptions of the various archetypes considered to be blocks of flats and non-blocks can be found in [Table 1](#table-1-manual-labels). Features are generated per building from building footprint and UPRN geodata.

## Contents

- [Scripts](#scripts)
- [i. Labelling individual properties as flats](#i-labelling-individual-properties-as-flats)
- [ii. Sampling data for manual labelling](#ii-sampling-data-for-manual-labelling)
- [iii. Manual labelling](#iii-manual-labelling)
- [iv. Model training and evaluation](#iv-model-training-and-evaluation)
- [v. Model application](#v-model-application)
- [vi. Caveats of block of flats classification methodology](#vi-caveats-of-block-of-flats-classification-methodology)
- [Features](#features)

## Scripts

The sampling, training and prediction steps are run in this order. Sampling and training only need to be run when the
model is retrained, for example when there is new labelled data or the method changes. Every pipeline run then loads an
existing trained model using `--model_release` (see [`pipeline/README.md`](../../README.md)).

| Script                                               | What it does                                                                                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`sample_for_labelling.py`](sample_for_labelling.py) | Draws a stratified sample of buildings containing flats and saves it as one KML file per labeller, for manual labelling.                               |
| [`labelled_data.py`](labelled_data.py)               | Loads the labelled KML files, groups the labels into block / non-block, resolves disagreements between labellers and assigns the train/test split.     |
| [`feature_engineering.py`](feature_engineering.py)   | Generates the features used by the model from building footprints and UPRN locations.                                                                  |
| [`train_model.py`](train_model.py)                   | Trains and evaluates the classifier, and contains the functions that predict with it and add the predicted label to UPRNs (used by `add_features.py`). |

## i. Labelling individual properties as flats

The processes of both feature engineering for the model and sampling for manual labelling require individual domestic UPRNs to be labelled with a boolean flag indicating whether they represent individual flats or not. For the purposes of this work, flats may be located within purpose-built blocks within converted houses or other buildings. A UPRN is considered to represent a flat if it is labelled by our pipeline as domestic and is located at the exact same coordinates as any other domestic UPRN. This is based on the assumption that UPRNs in the exact same location share an address point (e.g. a front door) and so are most likely to represent flats.

This is done by [`property_type.impute_set_flat_properties`](../../impute/property_type.py).

## ii. Sampling data for manual labelling

A sample of approximately 3000 buildings from the 2026 April OS OpenMap Local building footprints was taken for manual labelling. Buildings were joined to the UPRNs they contain, and the number of UPRNs labelled as a flat was calculated per building.

The aim was to identify which buildings containing flats were blocks of flats. As such, the sampling population consisted of buildings in Great Britain containing at least two flats, as buildings with no flats are considered non-blocks by default.

We aimed to generate a labelled dataset that represents the diversity of buildings containing flats across Great Britain. We therefore selected a set of primary strata and secondary constraints to sample against which we assumed would give us a representative sample of the true population.

Train/validation and test sets of data for the model were created up front in the sampling process in order to train the model for real-world application.

The full sample was created in the following steps:

1. Building footprints were enriched with the following information:
   - Building level attributes:
     - Area: London; rest of England; Scotland; Wales
     - Number of flats in footprint: 2-6; 7-15; 16+
     - Proportion of UPRNs in the footprint that are flats: >80%, <=80% binary
   - LSOA/ Data Zone level attributes:
     - Rurality: urban/rural binary
     - IMD decile group: high deprivation (deciles 1-3), not high deprivation (4-10) binary
2. Buildings were grouped into 24 primary strata to stratify sampling on according to area; number of flats; and deprivation binary. The remaining binary attributes were applied as secondary constraints: urban/rural binary; and proportion of flats binary.
3. A sample was then built to meet the following conditions:
   - Total sample size approximately equal to 3000 footprints. 75% (N~2250) of these were to form the training and validation set, and 25% (N~750) were to form the final hold-out test set. The training and validation set and the test sets were sampled separately using different methodologies.
   - To generate the training and validation sample:
     - 50 buildings were taken from each of the 24 primary strata first, with secondary constraints applied evenly (a 50/50 split across each binary attribute). This "padding" ensures smaller strata are still represented in model training.
     - The remaining buildings were then sampled proportionally to the total population across primary strata, with secondary constraints reflecting whole-population proportions.
     - The result is a sample that's close to whole-population proportions, but with extra representation for smaller groups.
   - We generated the hold-out test sample separately and entirely proportionally - i.e. both primary strata and secondary constraints reflect whole-population proportions with no padding.
4. The final dataset for manual labelling contained the full ~3000 building footprint sample (train & validation and test combined).

### How to run the sampling

[`sample_for_labelling.py`](sample_for_labelling.py) carries out the sampling above.

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
| `--release_date`                | No       | `YYYYMMDD` release of the domestic UPRNs (pipeline stage 1) to use. Also names the directory the outputs are saved to. Defaults to today.             |
| `--seed`                        | No       | Random seed. Defaults to 7.                                                                                                                           |
| `--target_n`                    | No       | Target number of buildings in the sample. Defaults to 3000.                                                                                           |
| `-m`, `--map_uprns_to_building` | No       | Generate the UPRN to building ID mapping from scratch and save it. Needs to be set the first time, because the training step reads the saved mapping. |
| `--save`                        | No       | Save outputs to S3, including the sample and the KML files for labelling. Without this, nothing is saved.                                             |

Outputs are saved to `outputs/models/block_of_flats_classifier/<release_date>/` on S3 (paths are set in
[`config/base.yaml`](../../../config/base.yaml)):

- The unlabelled sample, as a parquet file, whose file name includes the sample size and seed.
- One KML file per labeller, containing the building footprints that labeller has been assigned. 30 of each labeller's buildings (by default) are also assigned to a second labeller, so that agreement between labellers can be checked. Each building has a Google Maps link to its location and its `building_id`, number of flats and number of UPRNs in the description.
- The UPRN to building ID mapping (only with `--map_uprns_to_building`) and the enriched buildings, with and without geometries.

If a building was labelled in an earlier round of labelling, that label is carried over to the new sample. These need to be checked, confirmed or reassigned, because the sampling strategy has changed since that round.

## iii. Manual labelling

Through a previous labelling process, as well as stakeholder input, we generated a set of 24 different labels to use in manual labelling, representing the variety of archetypes we see in building footprints containing flats.

Labelling of the 3000 samples was performed by a group of three human labellers using the following information for each building:

- Count of UPRNs in the building footprint
- Count of UPRNs labelled as 'flats' in the building footprint
- Shape of the building footprint
- Satellite-, 3D-, and street-view of the building and surrounding area

Labellers were given descriptions and images of each identified archetype, and guidance and rules for labelling buildings consistently.

During labelling, buildings were given one of the 24 codes shown in the table below, which, in turn, are grouped into either 'block of flats', 'non-block', or 'excluded'. Labellers also assigned a confidence of either 1 (confident) or 0 (not confident) to their labels.

### Table 1. Manual labels

Contains the different possible labels that could be given to building footprints during manual labelling of footprints for block of flats classification.

| Label | Class          | Description (abbreviated)                                                                                                                                                                 |
| ----- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TB    | Block of flats | Tower blocks and high-rises.                                                                                                                                                              |
| BF    | Block of flats | Purpose-built block of flats including 1930s Art Deco blocks; deck-access estates; modern and new-builds; and social housing blocks.                                                      |
| BP    | Block of flats | Same as above but the building footprint represents a small section of a bigger block. I.e. this represents a section of a BF with <= 6 flats in it.                                      |
| IW    | Block of flats | Industrial warehouse (or similar) conversions into flats.                                                                                                                                 |
| TN    | Block of flats | Traditional Scottish tenements.                                                                                                                                                           |
| TC    | Block of flats | Large terrace to block of flat conversion.                                                                                                                                                |
| SF    | Block of flats | Sprawling mansion block e.g. a large old building converted into flats. These will usually be large and sprawling buildings with large garden(s) and separate entrances.                  |
| TT    | Block of flats | A mixture of a block of flats and another built form combined into the same building footprint. E.g. Shopping centre with flats on one side; or terrace with a block of flats on one end. |
| AS    | Block of flats | 3+ storeys of flats above shops.                                                                                                                                                          |
| NH    | Block of flats | Small number of flats across more than 2 storeys.                                                                                                                                         |
| TS    | Non-block      | 1-2 storeys of flats above shops.                                                                                                                                                         |
| WB    | Non-block      | A small wide block of flats. These are no more than 2 storeys tall (including ground floor). They can be purpose-built flats or conversions.                                              |
| TE    | Non-block      | Terraced housing. These could contain a mix of flats and houses, or just flats.                                                                                                           |
| TY    | Non-block      | Tyneside flats. Paired terraced houses designed specifically so that each building contains two flats.                                                                                    |
| BB    | Non-block      | Back-to-backs. Terraced houses built sharing back and sides with no rear outdoor space.                                                                                                   |
| MT    | Non-block      | Multiple semi-detached building footprints merged into one.                                                                                                                               |
| MA    | Non-block      | Purpose-built tiny block of 1-2 flats.                                                                                                                                                    |
| TF    | Non-block      | Terraced front, flats at the back. These are terraces on one side with attached 1- or 2-storey flats with separate entrances at the back.                                                 |
| FC    | Non-block      | Old buildings, like farm houses, schools, or churches, converted into homes.                                                                                                              |
| RA    | Non-block      | Retirement or sheltered housing with communal facilities and support services.                                                                                                            |
| OF    | Non-block      | A building footprint that doesn't match any of the archetypes, is not a block, and contains domestic properties. E.g. semi-detached houses; active farms; etc.                            |
| CO    | Excluded       | Contains non-domestic properties only e.g. shops; industrial estate buildings, etc.                                                                                                       |
| DE    | Excluded       | Derelict building.                                                                                                                                                                        |
| UL    | Excluded       | Unable to label. This label is used either when the building type is unclear OR when there is a clear error in the number of UPRNs assigned to that footprint.                            |

The 24 categories help to ensure a variety of built forms are captured in the labelled sample, and help to identify more specifically where the model is performing poorly.

The subsequent binary classification is ultimately based on how these different building types should be treated in the decision tree to assign a technology to, whether that be a communal solution versus a networked heat pump or individual solution.

Each labeller was given 30 samples which were independently labelled by another labeller to assess general alignment. Average agreement across the binary classification was 84%.

Manually labelled buildings which met the following criteria were excluded from the final training dataset:

- Buildings with a manual label confidence of 0
- Buildings which were labelled with 'UL' (unable to label), 'DE' (derelict), or 'CO' (non-domestic)
- Buildings with two labellers where labellers didn't agree at the binary level

The final dataset used for training contains 2604 buildings.

### How to record and save labels

Each labeller opens their KML file and renames each building polygon to its two-letter label followed by the confidence digit, for example `TB1`. The `building_id` in the polygon description must be kept, because it is used to link the label back to the building.

The labelled KML files are saved to the `labelled_data/` directory of the block of flats classifier directory on S3 (`labelled_dir` in [`config/base.yaml`](../../../config/base.yaml)). All `.kml` files in that directory are loaded for training. [`labelled_data.py`](labelled_data.py) then loads them, applies the exclusions above, prints the agreement between labellers and adds the train/test split from the sample. The groups of labels are set by `BLOCK_OF_FLATS_ARCHETYPES`, `NOT_BLOCKS_ARCHETYPES` and `EXCLUDED_ARCHETYPES` in that file.

## iv. Model training and evaluation

A random forest binary classifier model was trained to label building footprints as either blocks of flats or not. A random forest classifier was selected as it improved the initial F1 score compared to a baseline logistic regression model.

Features are generated per building from building footprint and UPRN geodata. See the [Features](#features) section for a full list of features.

The training data was used for hyperparameter tuning and cross-validation with sklearn's HalvingRandomSearchCV function, with StratifiedKFold employed for cross-validation. Hyperparameters were tuned to optimise for a best F1 score on the positive class ('blocks of flats').

The resulting best hyperparameters were used to train a final Random Forest Classifier model, which was evaluated on the hold-out test set. A final F1 score for the positive class was produced (Table 2).

### Table 2. Evaluation metrics

Evaluation metrics for random forest binary classifier during hyperparameter tuning and cross-validation and testing.

| Metric                                       | Value |
| -------------------------------------------- | ----- |
| Train set best F1 score                      | 0.71  |
| Test set F1 score                            | 0.71  |
| Precision on positive class (block of flats) | 0.64  |
| Recall on positive class                     | 0.79  |

The final model outputs both the predicted class and the probability of the class label for each building.

### How to run the training

[`train_model.py`](train_model.py) generates the features for the labelled buildings, then trains and evaluates the model.

```bash
python asf_heat_pump_suitability/pipeline/model/block_of_flats/train_model.py \
    --local_authorities GB \
    --model_release YYYYMMDD --uprn_release YYYYMMDD --save
```

| Argument              | Required | Description                                                                                                                                                       |
| --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--local_authorities` | Yes      | The local authorities the labelled sample was drawn from.                                                                                                         |
| `--model_release`     | No       | `YYYYMMDD` date of the sample and model data. Used to find the unlabelled sample on S3, and names the directory the model is saved to. Defaults to today.         |
| `--uprn_release`      | No       | `YYYYMMDD` release of the domestic UPRNs output (pipeline stage 1) to use. Defaults to today.                                                                     |
| `--save`              | No       | Save the labelled data, the model data (labelled data with features) and the trained model to S3. Without this, the model is trained and evaluated but not saved. |

The sample, enriched building footprints and UPRN to building ID mapping must already have been generated by `sample_for_labelling.py` for the same `--model_release`. The unlabelled sample is found using its file name, which includes the sample size and seed. These are currently set in `train_model.py` (`l=3005`, `seed=7`), and need to be updated there if the sample was generated with different values.

The script prints the best cross-validation F1 score and its spread across folds, the feature importances, the hold-out test F1 score, a classification report and a confusion matrix. The hyperparameter search space is set by `PARAM_DISTRIBUTIONS` in `train_model.py`, and the random state is fixed with `RANDOM_STATE`.

With `--save`, the following are saved to S3:

- The processed labelled data (`manually_labelled_block_of_flats`), which `add_features.py` reads.
- The model data: the labelled data joined to its features (`model_data`).
- The trained model as a pickle (`block_of_flats_building_classifier.pkl`).

## v. Model application

The trained model is applied to the subset of buildings containing any domestic UPRNs and at least one flat to classify them into 'block of flats' and 'non-blocks'. All buildings containing any domestic UPRNs with zero flats are assigned the label 'non-block'.

A corresponding boolean label is then assigned to each individual UPRN to indicate whether it is located within a block of flats or not, by using the class of the building footprint within which the UPRN is located.

This happens in pipeline stage 2 ([`add_features.py`](../../run/add_features.py)), which loads the trained model using `--model_release`. `predict_class_block_of_flats` classifies the buildings, and `extend_df_in_block_of_flats_label` adds the label to each UPRN as `in_block_of_flats`. Buildings which were manually labelled keep their manual label, with a probability of 1.0. Buildings with no flats have a null probability.

## vi. Caveats of block of flats classification methodology

The 'ground truth' manual labelling is itself noisy. Manual labelling of buildings was subjective and complex and was conducted by multiple independent manual labellers. Building forms seen in the sample were extremely variable and as such, there is some disagreement in labelling between labellers. Buildings were labelled primarily according to visual inspection, using available visual information such as signage, number of bins and doorbells, and visual appearance of buildings. Images of the buildings were not standardised across labellers (i.e. labellers could freely use Google Maps satellite and 3D view from as many angles as deemed necessary for labelling) so there was also variation in the information used to make a judgement across labellers. Due to these factors, it's likely that some buildings have been erroneously mislabelled in the training dataset.

Labellers were not given information about the spatial distribution of UPRNs within buildings and therefore had to make assumptions about this during labelling based on other available information. The result is that ambiguous archetypes, which tend to present with atypical storey counts and UPRN counts (e.g. the BP, NH, and TC labels), potentially are more often erroneously labelled than more typical archetypes like tower blocks (TB) and terraces (TE).

Building footprints limit our classification capacity as they can contain multiple built forms and our model operates at building footprint level. This means that some individual UPRNs within a given building footprint may receive a binary classification that is inaccurate for them individually, but appropriate for other UPRNs within the same footprint. For example, our model treats TT types (a single building footprint containing a combination of a block of flats and another built form) as a 'block of flats'. Future work may improve this and allow distinction of classifications within building footprints.

Building footprints may represent a smaller section of a larger building, as in the case of BP types (representing a smaller section of a bigger block of flats). These sections may exhibit similar features to building types in a different binary class, and therefore may affect the accuracy of the model and the final classification of UPRNs.

## Features

The features are generated per building in [`feature_engineering.py`](feature_engineering.py), from the building footprint and the UPRNs inside it. The list of features used by the model is `FEATURES` in [`train_model.py`](train_model.py).

| Group                      | Feature                                                             | Description                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Building footprint         | `n_UPRNs`, `n_flats`, `n_domestic_uprns`                            | Count of all UPRNs, flats and domestic UPRNs in the building.                                                                    |
|                            | `building_area_m2`, `building_perimeter_m`                          | Area and perimeter of the building footprint.                                                                                    |
|                            | `proportion_flats`, `domestic_proportion_flats`                     | Proportion of all UPRNs, and of domestic UPRNs, that are flats.                                                                  |
|                            | `UPRNs_per_building_m2`                                             | UPRNs per m2 of building footprint.                                                                                              |
| UPRN spread (concave hull) | `concave_hull_area_m2`                                              | Area of the concave hull of the UPRN points in the building, which represents how spread out the UPRNs are within the footprint. |
|                            | `uprns_per_hull_area_m2`, `flats_per_hull_area_m2`                  | UPRNs and flats per m2 of concave hull. Set to -1 when all UPRNs share a location (hull area is 0).                              |
| Stacked UPRNs              | `avg_n_stacked_uprns`, `std_n_stacked_uprns`, `max_n_stacked_uprns` | Mean, standard deviation and maximum number of UPRNs sharing exactly the same coordinates in the building.                       |
| Nearest neighbour          | `n_flats_nearest`, `nn_distance_m`                                  | Number of flats in the nearest building containing flats, and distance to it. Set to -100 if there is no neighbour within 30m.   |
