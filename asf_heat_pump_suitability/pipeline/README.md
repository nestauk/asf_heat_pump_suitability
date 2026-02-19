# asf_heat_pump_suitability/pipeline/

Pipeline entry scripts and supporting modules for the heat pump suitability pipeline.

## Structure

```
asf_heat_pump_suitability/pipeline/
├── uprns.py                       Step 1: filter residential UPRNs (ahps-uprns)
├── add_features.py                Step 2: add flat flag and outdoor space (ahps-add-features)
├── setup/
│   ├── stream_inspire_files.py    One-time: stream INSPIRE data to S3 (ahps-stream-inspire)
│   └── train_model.py             One-time: train block-of-flats classifier (ahps-train-model)
├── model/
│   └── block_of_flats/
│       └── feature_engineering.py Block-of-flats model feature engineering
├── impute/
│   └── property_type.py           Flat detection via shared-coordinate heuristic
├── prepare_features/              Feature preparation modules (EPC enrichment)
├── run/
│   └── heat_network_city_centres.py  Labels UPRNs in heat network zones / city centres
└── transform/                     Core geospatial data transformations
    ├── uprns.py                   UPRN coordinate generation and filtering
    ├── outdoor_space.py           INSPIRE-based outdoor space estimation
    ├── non_residential_entities.py Non-residential building classification
    └── poi.py                     Point-of-interest transformations
```

## Main pipeline steps

### Step 1: Filter residential UPRNs — `uprns.py`

```bash
uv run ahps-uprns --area plymouth
```

Accepted `--area` values: `plymouth`, `plymouth_similar`, `sampling`, `gb`.

Output: `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{area}_residential_uprns.parquet`

### Step 2: Add features — `add_features.py`

```bash
uv run ahps-add-features --uprns s3://asf-heat-pump-suitability/.../plymouth_residential_uprns.parquet
```

Output: `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{stem}_with_features.parquet`

## Setup scripts (run once)

### Stream INSPIRE land registry files

```bash
uv run ahps-stream-inspire --nations ew   # England & Wales
uv run ahps-stream-inspire --nations s    # Scotland
uv run ahps-stream-inspire --nations all  # All nations
```

### Train block-of-flats classifier

```bash
uv run ahps-train-model \
    --uprns s3://asf-heat-pump-suitability/.../domestic_uprns.parquet \
    --labelled_data s3://asf-heat-pump-suitability/.../labelled_buildings.parquet
```

## Cloud execution

```bash
orbit launch --script asf_heat_pump_suitability/pipeline/uprns.py \
    --team <team> --project asf_heat_pump_suitability -- --area gb

orbit launch --script asf_heat_pump_suitability/pipeline/add_features.py \
    --team <team> --project asf_heat_pump_suitability -- --uprns s3://...
```
