# Current pipeline

## Base input: UPRNs with geospatial coordinates

## Main output: UPRNs with features required for decision tree

To run the current pipeline:

First run the `uprns.py` script to filter UPRNs to those which we think are domestic only. This outputs a file to S3 with domestic UPRNs.
Then run the `add_features.py` script using the domestic UPRNs file as input. This adds all the features required for the decision tree. The output is one row per UPRN with all the features required.

Companion scripts - these need to be run to generate models / data that are used in the main pipeline but don't need to be rerun every time the pipeline is run:

`train_model.py` - This trains the classification model for blocks of flats. The model is loaded and called in add_features.py (currently unmerged).
`run_stream_inspire_files.py` - this is an old script from V1.0.0 but is required to save land registry data to S3 which is used to generate outdoor space estimates in the add_features.py script. This streams land registry files directly from the government websites to S3.

The subdirectories we are using in the latest version within /pipeline are:

`/impute/`
`/prepare_features/` - for some legacy code only
`/run/`
`/transform/`

# Request

I want to refactor this repository to:

- Cut out extraneous files not used
- Move to `uv`
- Move to Python 3.12
- Have sufficient tests across our modules, unit and smoke
- Have orchestration to easily re-run pipelines as necessary
- Ensure that running locally doesn't accidentally overwrite S3
- Enalbe easy REPL dev/testing locally
- Is sufficiently well documented
- Easy to update and maintain for a data scientist, not a data engineer
