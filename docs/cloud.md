# Cloud Deployment

The pipeline runs on the cloud via [arm_orbit](https://github.com/nestauk/arm_orbit), Nesta's tool for running Python scripts on AWS Batch (Fargate).

## How arm_orbit Works

arm_orbit auto-detects `pyproject.toml` at the repo root, installs the package via `uv pip install .` inside a Docker container, and submits the script to AWS Batch.

## Running on the Cloud

```bash
# Step 1: Filter UPRNs
orbit launch --script pipeline/uprns.py --team <team> --project <project>

# Step 1: Specific area
orbit launch --script pipeline/uprns.py --team <team> --project <project> -- --area gb

# Step 2: Add features
orbit launch --script pipeline/add_features.py --team <team> --project <project> \
  -- --uprns s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns.parquet
```

## Environment Variables

| Variable           | Purpose                                     | Default                     |
| ------------------ | ------------------------------------------- | --------------------------- |
| `DATA_MODE`        | `s3` or `local`                             | `s3`                        |
| `AWS_ENDPOINT_URL` | Override S3 endpoint (set by moto in tests) | unset                       |
| `S3_BUCKET`        | Override default S3 bucket                  | `asf-heat-pump-suitability` |
| `ORBIT_ENV`        | arm_orbit environment (dev/staging/prod)    | `prod`                      |

## Companion Scripts

Run these once before the main pipeline (or whenever source data changes):

```bash
# Ingest INSPIRE land registry data to S3
orbit launch --script pipeline/setup/stream_inspire_files.py \
  --team <team> --project <project> -- -n all

# Train the block-of-flats classifier
orbit launch --script pipeline/setup/train_model.py \
  --team <team> --project <project> \
  -- --uprns s3://... --labelled_data s3://...
```
