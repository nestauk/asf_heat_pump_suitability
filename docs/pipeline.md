# Pipeline

## Architecture

The pipeline consists of two independent CLI scripts that communicate via S3. Each step reads its input from and writes its output to config-defined S3 paths — no `--input-path` argument is passed between steps.

```
pipeline/uprns.py  →  (S3)  →  pipeline/add_features.py
```

## Step 1: Filter UPRNs (`pipeline/uprns.py`)

Reads the OS Open UPRN dataset from S3, filters to residential properties using:

1. Building footprint spatial join (OS OpenMap Local)
2. Non-residential building exclusion (important buildings, railway stations, POI)
3. EPC register lookups (domestic = include, commercial = exclude)

**Output:** `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{area}_residential_uprns.parquet`

### Area options (`--area`)

| Flag               | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `plymouth`         | Plymouth Local Authority only (default for dev/test)       |
| `plymouth_similar` | Plymouth + Liverpool, Portsmouth, Southampton, Swansea     |
| `sampling`         | Plymouth + Bath, Bradford, Glasgow, Manchester, Nottingham |
| `gb`               | Full Great Britain                                         |

## Step 2: Add Features (`pipeline/add_features.py`)

Reads the domestic UPRN dataset produced by step 1 and adds:

1. **Flat classification** (`property_type_flat`) — geometric imputation based on shared UPRN coordinates
2. **Outdoor space** (`max_contiguous_outdoor_space_area_m2`, `total_outdoor_space_area_m2`) — from INSPIRE land registry and OS building footprints

**Output:** `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{stem}_with_features.parquet`

## Running the pipeline

```bash
# Step 1 (Plymouth, default)
python pipeline/uprns.py

# Step 1 (specific area)
python pipeline/uprns.py --area plymouth_similar

# Step 2
python pipeline/add_features.py --uprns s3://asf-heat-pump-suitability/.../plymouth_residential_uprns.parquet
```
