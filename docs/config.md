# Configuration

## Overview

Configuration is loaded from `asf_heat_pump_suitability/config/base.yaml` and exposed via two mechanisms:

1. **Dict access** (`asf_heat_pump_suitability.config`) — for backwards compatibility with existing utility functions
2. **Pydantic settings** (`asf_heat_pump_suitability.config.settings.load_settings()`) — for new pipeline entry scripts; supports environment variable overrides

## Environment Variable Overrides

Any Pydantic field can be overridden by an environment variable:

| Variable           | Purpose                              | Default                     |
| ------------------ | ------------------------------------ | --------------------------- |
| `DATA_MODE`        | `s3` or `local`                      | `s3`                        |
| `AWS_ENDPOINT_URL` | Override S3 endpoint (moto in tests) | unset                       |
| `S3_BUCKET`        | Override default S3 bucket           | `asf-heat-pump-suitability` |
| `ORBIT_ENV`        | arm_orbit environment                | `prod`                      |

## Key Config Sections

### `data.geodata`

S3 paths to geospatial input data (OS Open UPRN, OS OpenMap Local, boundaries, spatial signatures).

### `data.epc`

S3 paths to EPC registers (domestic and commercial).

### `data.processed`

S3 paths to preprocessed inputs (POI categories, cached UPRN datasets).

### `constant.grid_squares`

OS National Grid 100km square codes for each geographic area:

| Area                      | Grid squares                 |
| ------------------------- | ---------------------------- |
| `plymouth`                | `SX`                         |
| `plymouth_similar_cities` | `SD, SJ, SN, SS, SU, SX, SZ` |
| `sampling_areas`          | `NS, SD, SE, SJ, SK, ST, SX` |

### `output`

S3 path templates for pipeline outputs:

- `residential_uprns_template` — output of `pipeline/uprns.py`
- `features_template` — output of `pipeline/add_features.py`
- `save_as.block_of_flats_model` — trained model pickle path

### `mapping`

Data mappings used in feature preparation:

- `build_year_pre_cols` / `build_year_post_cols` — EPC build year band groupings
- `pre_post_1930_epc` — EPC construction period to pre/post-1930 mapping
- `ruc_two_fold` — Rural-Urban Classification to two-fold (Urban/Rural)
- `ruc_EW_ten_fold` / `ruc_S_eight_fold` — detailed RUC classifications
