# pipeline/

Top-level CLI entry scripts for the heat pump suitability pipeline. These scripts are the main
execution points — they import from the installable `asf_heat_pump_suitability` package and are run
directly with `uv run python` or launched on AWS Batch via
[arm_orbit](../docs/cloud.md).

## Main pipeline steps

### Step 1: Filter residential UPRNs — `uprns.py`

Loads OS Open UPRN data for a geographic area, identifies residential properties using building
footprints and EPC data, and writes a filtered parquet to S3.

```bash
uv run python pipeline/uprns.py --area <area>
```

| Argument | Required | Values                                           | Description                |
| -------- | -------- | ------------------------------------------------ | -------------------------- |
| `--area` | yes      | `plymouth`, `plymouth_similar`, `sampling`, `gb` | Geographic area to process |

Output: `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{area}_residential_uprns.parquet`

### Step 2: Add features — `add_features.py`

Takes the filtered residential UPRNs from Step 1 and enriches each property with:

- `property_type_flat` — whether the property is a flat / in a block of flats (geometric imputation)
- `max_contiguous_outdoor_space_area_m2` — largest contiguous outdoor area (from INSPIRE polygons)
- `total_outdoor_space_area_m2` — total outdoor area within the land parcel

```bash
uv run python pipeline/add_features.py --uprns <path>
```

| Argument  | Required | Description                                        |
| --------- | -------- | -------------------------------------------------- |
| `--uprns` | yes      | Path to parquet from Step 1 (S3 URI or local path) |

Output: `s3://asf-heat-pump-suitability/local_heat_planning/outputs/{stem}_with_features.parquet`

## Setup scripts (run once)

These scripts must be run before the main pipeline when setting up a new environment or when source
data needs refreshing.

### `setup/stream_inspire_files.py`

Streams INSPIRE land registry XML files from the government data service and saves them (unzipped)
to S3. Required before running `add_features.py`.

```bash
uv run python pipeline/setup/stream_inspire_files.py --nations ew   # England & Wales
uv run python pipeline/setup/stream_inspire_files.py --nations s     # Scotland
uv run python pipeline/setup/stream_inspire_files.py --nations all   # All nations
```

### `setup/train_model.py`

Trains the block-of-flats building classifier used by `add_features.py` to identify flats. Saves
the fitted model as a pickle file to S3.

```bash
uv run python pipeline/setup/train_model.py
```

The model requires pre-processed training data in S3. See the script docstring for details.

## Cloud execution

To run either main step on AWS Batch via arm_orbit:

```bash
orbit launch --script pipeline/uprns.py --team <team> --project asf_heat_pump_suitability -- --area gb
orbit launch --script pipeline/add_features.py --team <team> --project asf_heat_pump_suitability -- --uprns s3://...
```

See [docs/cloud.md](../docs/cloud.md) for full cloud setup instructions.
