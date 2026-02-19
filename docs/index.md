# ASF Heat Pump Suitability

A pipeline to generate geospatial features required for the heat pump suitability decision tree.

## Overview

The pipeline takes UK domestic UPRNs with geospatial coordinates as input and outputs UPRNs enriched with features used in the heat pump suitability decision tree, including:

- **Flat / apartment classification** — geometric and ML-based classification of UPRNs in blocks of flats
- **Outdoor space estimates** — max contiguous and total outdoor space per land parcel (m²) derived from OS OpenMap Local building footprints and INSPIRE land registry data

## Pipeline Steps

| Step | Script                     | Description                                               |
| ---- | -------------------------- | --------------------------------------------------------- |
| 1    | `pipeline/uprns.py`        | Filter OS Open UPRNs to domestic (residential) properties |
| 2    | `pipeline/add_features.py` | Add geospatial features to domestic UPRNs                 |

## Companion Scripts (run once)

| Script                                   | Description                                                       |
| ---------------------------------------- | ----------------------------------------------------------------- |
| `pipeline/setup/stream_inspire_files.py` | Stream INSPIRE land registry files from government websites to S3 |
| `pipeline/setup/train_model.py`          | Train the block-of-flats classifier and save to S3                |

## Technology Stack

- **Python 3.12** with **uv** for package management
- **Polars** for tabular data processing
- **GeoPandas** / **Shapely** for geospatial operations
- **Pydantic-settings** for typed configuration
- **moto** for S3 mocking in tests
- **arm_orbit** for cloud execution (AWS Batch via Fargate)
