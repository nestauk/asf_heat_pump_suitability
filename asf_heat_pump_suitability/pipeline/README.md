# asf_heat_pump_suitability/pipeline

## Directory structure

```
asf_heat_pump_suitability/pipeline/
├── uprns.py                  CLI: filter all UPRNs to domestic-only
├── add_features.py           CLI: add features required for the decision tree
├── run.py                    CLI: orchestrate both steps end-to-end
├── impute/
│   └── property_type.py      Impute flat/apartment property type flag
├── model/
│   └── block_of_flats/
│       ├── feature_engineering.py  Generate building-level features for classifier
│       └── train_model.py          Train/evaluate Random Forest block-of-flats classifier
├── setup/
│   ├── train_model.py        CLI: train and save block-of-flats model to S3
│   └── stream_inspire_files.py  CLI: stream INSPIRE land registry files to S3
└── transform/
    ├── non_residential_entities.py  Flag non-residential UPRNs
    ├── outdoor_space.py             Estimate outdoor space from land parcels
    ├── poi.py                       Transform Points of Interest data
    └── uprns.py                     Generate UPRN geodataframes and building mappings
```

## Entry-point scripts

### `uprns.py` — filter UPRNs to domestic

Reads OS Open UPRN and EPC registers; outputs a parquet of domestic-only UPRNs with coordinates.

```bash
uv run python asf_heat_pump_suitability/pipeline/uprns.py --local-authorities plymouth
```

### `add_features.py` — add decision-tree features

Takes `domestic_uprns.parquet` and adds `property_type_flat`, `in_block_of_flats`,
`max_contiguous_outdoor_space_area_m2`, and `total_outdoor_space_area_m2`.

```bash
uv run python asf_heat_pump_suitability/pipeline/add_features.py --local-authorities plymouth
```

### `run.py` — full pipeline

Runs `uprns.py` then `add_features.py` in sequence.

```bash
uv run python asf_heat_pump_suitability/pipeline/run.py --local-authorities plymouth
```

## Setup companion scripts

These are run once (or when source data changes), not as part of the regular pipeline.

### `setup/train_model.py` — train block-of-flats classifier

Trains the Random Forest classifier used by `add_features.py`. Requires labelled training data
and a domestic UPRN parquet covering the same area.

```bash
uv run python asf_heat_pump_suitability/pipeline/setup/train_model.py \
    --uprns s3://asf-local-heat-planning-tool/outputs/sampling_areas_residential_uprns.parquet \
    --labelled-data s3://asf-local-heat-planning-tool/inputs/reference/manually_labelled_block_of_flats.parquet \
    --save
```

### `setup/stream_inspire_files.py` — stream INSPIRE land registry files to S3

Downloads INSPIRE land parcel polygons from HMLR (England & Wales) and Registers of Scotland
and streams them to the `asf-local-heat-planning-tool` S3 bucket. Run quarterly when data is updated.

```bash
# Both nations
uv run python asf_heat_pump_suitability/pipeline/setup/stream_inspire_files.py --nations all

# England & Wales only
uv run python asf_heat_pump_suitability/pipeline/setup/stream_inspire_files.py --nations ew

# Scotland only
uv run python asf_heat_pump_suitability/pipeline/setup/stream_inspire_files.py --nations s
```
