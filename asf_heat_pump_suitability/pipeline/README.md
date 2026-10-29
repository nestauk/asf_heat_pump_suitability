# Pipeline

This directory contains the code that turns raw input datasets into the final outputs of the project: for each local
authority, the most suitable low-carbon heating technology for every property, grouped into clusters of buildings, with
contextual information about each cluster.

Setup instructions (environment, AWS access) are in the [main README](../../README.md#setup). All commands below are
run from the repository root, and read from and save to S3.

## Contents

- [How the pipeline fits together](#how-the-pipeline-fits-together)
- [1. Run the whole pipeline](#1-run-the-whole-pipeline)
  - [Option A: `pipeline.py` (locally)](#option-a-pipelinepy-locally)
  - [Option B: `orbit_pipeline.py` (remotely, in batches)](#option-b-orbit_pipelinepy-remotely-in-batches)
- [2. Run each stage individually](#2-run-each-stage-individually)
- [3. Compare two versions of an output (`compare_versions.py`)](#3-compare-two-versions-of-an-output-compare_versionspy)
- [4. Optimise input datasets (`optimise_inputs.py`)](#4-optimise-input-datasets-optimise_inputspy)
- [5. Train the block of flats model (`train_model.py`)](#5-train-the-block-of-flats-model-train_modelpy)

## How the pipeline fits together

```
                     ┌─ one-off / occasional set up ─┐
                     │  optimise_inputs.py           │  speeds up loading of large input datasets
                     │  train_model.py               │  trains the block of flats classifier
                     └───────────────┬───────────────┘
                                     ▼
 uprns.py → add_features.py → decision_tree.py → cluster.py → compute_contextual_features.py
                                                                          │
                                                                          ▼
                                                      compare_versions.py (optional check)
```

The five stages in the middle are what `pipeline.py` runs, in order, for each local authority. Each stage reads the
previous stage's output from S3, so they must be run in this order. Outputs are saved to a dated release directory
(`YYYYMMDD`) on S3.

| Stage | Script                                                                     | What it does                                                                                                                                                                      |
| ----- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | [`transform/uprns.py`](transform/uprns.py)                                 | Filters all OS UPRNs down to residential UPRNs.                                                                                                                                   |
| 2     | [`run/add_features.py`](run/add_features.py)                               | Adds features to each UPRN (flat flag, block of flats flag, listed building, off-gas, coastal, protected area, outdoor space, EPC-derived info).                                  |
| 3     | [`transform/decision_tree.py`](transform/decision_tree.py)                 | Applies the decision tree to give each UPRN its most suitable heating technology, then assigns one solution per building.                                                         |
| 4     | [`cluster/cluster.py`](cluster/cluster.py)                                 | Groups neighbouring buildings with the same technology (not separated by barriers such as rivers or roads) into clusters.                                                         |
| 5     | [`run/compute_contextual_features.py`](run/compute_contextual_features.py) | Summarises each cluster (tenure, EPC ratings, outdoor space, listed buildings, solar PV, etc.) and creates the final GeoJSON, including heat network zone and city centre layers. |

## 1. Run the whole pipeline

### Option A: `pipeline.py` (locally)

[`run/pipeline.py`](run/pipeline.py) runs the five stages in sequence for each local authority you give it, and saves
every stage's outputs to S3. If a stage fails for a local authority, that local authority is skipped and the pipeline
moves on to the next one. At the end, any errors or warnings are written to a CSV log on S3.

```bash
python asf_heat_pump_suitability/pipeline/run/pipeline.py \
    --local_authorities 'Plymouth' 'Vale of Glamorgan' \
    --model_release YYYYMMDD
```

| Argument              | Required | Description                                                                                                                                                                                                                                                         |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--local_authorities` | Yes      | One or more local authority names (case insensitive).                                                                                                                                                                                                               |
| `--model_release`     | Yes      | `YYYYMMDD` date of the block of flats model training run (see [`train_model.py`](#5-train-the-block-of-flats-model-train_modelpy)). Used by `add_features.py` to load the trained classifier.                                                                       |
| `--release_date`      | No       | `YYYYMMDD` release directory to read from and save to. Defaults to today. It is pinned for all stages, so a run that crosses midnight still writes to one directory.                                                                                                |
| `--prod`              | No       | Push the final output to production (the staging area). Only use this when running from `dev`.                                                                                                                                                                      |
| `--compare`           | No       | After a local authority succeeds, compare each stage's output with its previous version (see [`compare_versions.py`](#3-compare-two-versions-of-an-output-compare_versionspy)). A failed comparison is listed at the end but does not fail the local authority.     |
| `--trigger`           | No       | Requires `--compare`. Why the run is being made: `methodology_change` (the code or method changed, the inputs did not) or `input_release` (a new release of an input dataset, the code did not change). This sets the tolerances the comparison is checked against. |
| `--old_release_date`  | No       | Requires `--compare`. Compare against this release (`YYYYMMDD`) instead of each stage's latest earlier version. Must be earlier than the release date.                                                                                                              |

### Option B: `orbit_pipeline.py` (remotely, in batches)

[`run/orbit_pipeline.py`](run/orbit_pipeline.py) is for running many local authorities at once. It splits the local
authorities into chunks and launches one [`orbit`](https://github.com/nestauk/orbit) job per chunk (8 CPUs, 60GB memory
each), where each job runs `pipeline.py` for that chunk's local authorities. When the jobs have been launched, it
combines the error logs from all chunks into a single file on S3.

`orbit` is for Nesta staff only, and needs to be installed with `make install install-orbit` (see the main README).

```bash
# Specific local authorities, run together as a single chunk
python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py \
    --local_authorities 'glasgow city' 'south lanarkshire' \
    --model_release YYYYMMDD

# All local authorities in Great Britain, in chunks of 20 local authorities each
python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py \
    --local_authorities GB --size 20 \
    --model_release YYYYMMDD

# All local authorities in Great Britain, split into 10 chunks
python asf_heat_pump_suitability/pipeline/run/orbit_pipeline.py \
    --local_authorities GB --size 10 --is_number_of_chunks \
    --model_release YYYYMMDD
```

It takes the same `--local_authorities`, `--model_release`, `--release_date`, `--prod`, `--compare`, `--trigger` and
`--old_release_date` arguments as `pipeline.py`, which are passed on to every chunk. It also has:

| Argument                 | Description                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| `--local_authorities GB` | Use `GB` instead of names to run every local authority in Great Britain.                     |
| `--size N`               | Number of local authorities in each chunk. If omitted, everything runs in a single chunk.    |
| `--is_number_of_chunks`  | Treat `--size` as the number of chunks rather than the size of each chunk.                   |
| `-o` / `--orbit_args`    | Extra arguments to pass to `orbit`, e.g. `-o on-demand` (a `--` prefix is added if missing). |

With `--prod`, the launcher also reads `front_end_s3_bucket` and `front_end_staging_s3_path` from your `.env` file and
passes them to the jobs, because `.env` is not uploaded to `orbit`.

## 2. Run each stage individually

You can run any stage on its own, for example to debug a stage or re-run it after a failure. Because each stage reads the
previous stage's output from S3, the earlier stages must already have been run for the same local authority and
`--release_date`.

All stages share these arguments:

- `--local_authorities`: one or more local authority names (case insensitive).
- `--save`: save outputs to S3. **Without this flag, outputs are not saved.** (`pipeline.py` always passes it.)
- `--release_date YYYYMMDD`: the release directory to read from and save to (defaults to today). Use the same date for
  every stage in a run.

### Stage 1: `uprns.py`

```bash
python asf_heat_pump_suitability/pipeline/transform/uprns.py \
    --local_authorities 'Plymouth' --save
```

A UPRN is treated as residential if it is in the domestic EPC register, or if it is inside a building footprint that is
not a non-residential building type and is not in the non-domestic EPC register.

### Stage 2: `add_features.py`

```bash
python asf_heat_pump_suitability/pipeline/run/add_features.py \
    --local_authorities 'Plymouth' --model_release YYYYMMDD --save
```

Needs `--model_release`: the date of the trained block of flats model (see [section 5](#5-train-the-block-of-flats-model-train_modelpy)).

### Stage 3: `decision_tree.py`

```bash
python asf_heat_pump_suitability/pipeline/transform/decision_tree.py \
    --local_authorities 'Plymouth' --save
```

The decision tree uses whether the property is in a city centre or planned heat network zone, its maximum contiguous
outdoor space, and whether it is in a block of flats.

### Stage 4: `cluster.py`

```bash
python asf_heat_pump_suitability/pipeline/cluster/cluster.py \
    --local_authorities 'Plymouth' --save
```

Saves the clusters and the anchor loads (high heat demand buildings such as schools and hospitals).

### Stage 5: `compute_contextual_features.py`

```bash
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py \
    --local_authorities 'Plymouth' --save
```

Extra arguments:

- `--detail {full,simplified}`: level of detail of the spatial signatures used to label city centres. Defaults to `full`.
- `--prod`: push the output to production (the staging area). Only use this when running from `dev`.

## 3. Compare two versions of an output (`compare_versions.py`)

[`validate/compare_versions.py`](validate/compare_versions.py) checks what changed between two dated releases of one
stage's output for one local authority. It is useful after changing the code or after a new input dataset release, to
check that the outputs have changed in the way you expected, and not in a way you did not.

The report covers row and UPRN counts, schema changes, UPRNs added or removed, per-technology counts and a technology
transition matrix (decision tree), cluster counts and area changes with distribution plots (cluster and contextual
features), UPRNs with no cluster, and the commits that touched the stage's code between the two versions. It writes a
markdown report, a JSON of the headline numbers and distribution plots to `outputs/comparisons/`, and logs a summary to
the console. With `--save`, these are also uploaded to the new release's `compare_versions/` folder on S3.

```bash
# Compare the latest two versions found on S3, with no tolerance checks
python asf_heat_pump_suitability/pipeline/validate/compare_versions.py \
    --stage decision_tree --local_authority plymouth

# Compare a new release against the latest earlier one
python asf_heat_pump_suitability/pipeline/validate/compare_versions.py \
    --stage decision_tree --local_authority plymouth \
    --new_release_date 20260722

# Compare two specific releases, and check them against a trigger's tolerances
python asf_heat_pump_suitability/pipeline/validate/compare_versions.py \
    --stage decision_tree --local_authority plymouth \
    --old_release_date 20260601 --new_release_date 20260722 \
    --trigger methodology_change
```

| Argument             | Required | Description                                                                                                                                                                       |
| -------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--stage`            | Yes      | The stage whose output to compare: `uprns`, `add_features`, `decision_tree`, `cluster` or `compute_contextual_features`.                                                          |
| `--local_authority`  | Yes      | A single local authority, e.g. `'East Lothian'` or `east_lothian`.                                                                                                                |
| `--new_release_date` | No       | The newer version, `YYYYMMDD`. On its own, it is compared with the latest earlier version. If there is no earlier version, nothing is written and the script exits without error. |
| `--old_release_date` | No       | The older version, `YYYYMMDD`. Must be given together with `--new_release_date`.                                                                                                  |
| `--trigger`          | No       | `methodology_change` or `input_release`. Checks the numbers against that trigger's tolerances. Without it, the numbers are shown with no pass/fail judgement.                     |
| `--report_dir`       | No       | Where the markdown report is written. Defaults to `outputs/comparisons/`.                                                                                                         |
| `--save`             | No       | Also upload the report, plots and JSON to S3.                                                                                                                                     |

You can also run this automatically for every stage by adding `--compare` to `pipeline.py` or `orbit_pipeline.py`.

## 4. Optimise input datasets (`optimise_inputs.py`)

[`run/optimise_inputs.py`](run/optimise_inputs.py) converts the large raw input datasets (UPRNs, UPRN lookup, points of
interest, domestic and commercial EPC, coastline) into more efficient parquet files and, where possible, splits
geospatial data into files by 100km British National Grid square (e.g. `SX`, `NT`). The pipeline can then load only the
grid squares a local authority covers, which saves a lot of time and memory. Outputs are saved to S3.

It only needs to be run when the raw input datasets are updated, not for every pipeline run.

```bash
# All datasets, all grid squares
python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py

# Specific grid squares only (e.g. to re-process or test)
python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py --grid_squares SX SY

# Specific datasets only
python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py --datasets EPC_domestic POI
```

| Argument         | Description                                                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--grid_squares` | 100km grid squares to process. Defaults to all grid squares in the data.                                                           |
| `--datasets`     | Datasets to process, from `UPRN`, `UPRN_lookup`, `POI`, `EPC_domestic`, `EPC_commercial` and `coastline`. Defaults to all of them. |

## 5. Train the block of flats model (`train_model.py`)

[`model/block_of_flats/train_model.py`](model/block_of_flats/train_model.py) trains the random forest classifier that
`add_features.py` uses to decide whether a building is a block of flats. The classifier uses features of the building
footprint and the UPRNs inside it, and is trained on buildings that have been labelled by hand. It saves the trained
model and the labelled data it was trained on to S3, in a directory named by `--model_release`.

You only need to retrain the model when the training data or method changes; the pipeline then uses a trained model
through `--model_release`.

```bash
python asf_heat_pump_suitability/pipeline/model/block_of_flats/train_model.py \
    --local_authorities 'Plymouth' 'Glasgow City' \
    --model_release YYYYMMDD --uprn_release YYYYMMDD --save
```

| Argument              | Required | Description                                                                                                                                               |
| --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--local_authorities` | Yes      | The local authorities the labelled sample was drawn from.                                                                                                 |
| `--model_release`     | No       | `YYYYMMDD` date of the model training run. It is used to find the unlabelled sample on S3, and is the directory the model is saved to. Defaults to today. |
| `--uprn_release`      | No       | `YYYYMMDD` release of the domestic UPRNs output (stage 1) to use. Defaults to today.                                                                      |
| `--save`              | No       | Save the trained model and the labelled data to S3. Without this, the model is trained and evaluated but not saved.                                       |

The training data needs to exist before the model is trained. The other scripts in
[`model/block_of_flats/`](model/block_of_flats/) create it:

- [`sample_for_labelling.py`](model/block_of_flats/sample_for_labelling.py) draws a sample of buildings containing flats
  and saves it for manual labelling.
- [`labelled_data.py`](model/block_of_flats/labelled_data.py) loads the manually labelled buildings and prepares them for
  training.
- [`feature_engineering.py`](model/block_of_flats/feature_engineering.py) builds the building-level features used by
  the classifier.
