# ASF Heat Pump Suitability

The `asf_heat_pump_suitability` pipeline identifies domestic properties across Great Britain and
computes features relevant to heat pump planning. Starting from the full OS Open UPRN dataset it
filters to residential properties, then enriches each property with:

- whether it is a flat / in a block of flats
- estimated outdoor space (contiguous garden area) from INSPIRE land registry polygons

Outputs are saved to S3 and are used as inputs to downstream heat-planning analyses.

Read more about the project [here](https://www.nesta.org.uk/project/mapping-heat-pump-suitability-across-great-britain/).

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repo
git clone https://github.com/nestauk/asf_heat_pump_suitability.git
cd asf_heat_pump_suitability

# Install the package and all dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

To verify the setup run the test suite (credential-free; S3 is mocked with moto):

```bash
uv run pytest tests/
```

## Running the pipeline

The pipeline has two independent steps, available as console scripts after `uv sync`:

**Step 1 — Filter residential UPRNs**

```bash
uv run ahps-uprns --area plymouth
```

Accepted `--area` values: `plymouth`, `plymouth_similar`, `sampling`, `gb`.

**Step 2 — Add features**

```bash
uv run ahps-add-features --uprns s3://asf-heat-pump-suitability/.../plymouth_residential_uprns.parquet
```

See [asf_heat_pump_suitability/pipeline/README.md](asf_heat_pump_suitability/pipeline/README.md)
for full argument documentation and the one-time setup scripts (`ahps-stream-inspire`,
`ahps-train-model`) that must be run before the main pipeline.

## Repository structure

```
asf_heat_pump_suitability/         Installable Python package
├── config/
│   ├── base.yaml                  Data source paths and constants
│   ├── settings.py                Pydantic-settings config model
│   └── README.md                  Data source citations and attributions
├── getters/
│   ├── base_getters.py            Generic S3 / HTTP loaders
│   ├── load_boundaries.py         Administrative boundary loaders
│   ├── load_geodata.py            Geospatial data loaders
│   └── load_tree_input.py         Tree-model input loaders
├── pipeline/
│   ├── uprns.py                   Step 1 entry point (ahps-uprns)
│   ├── add_features.py            Step 2 entry point (ahps-add-features)
│   ├── setup/
│   │   ├── stream_inspire_files.py  One-time INSPIRE streaming (ahps-stream-inspire)
│   │   └── train_model.py           One-time model training (ahps-train-model)
│   ├── model/block_of_flats/      Block-of-flats model feature engineering
│   ├── impute/                    Imputation logic (flat detection)
│   ├── prepare_features/          Feature preparation modules
│   ├── run/                       Orchestration helpers
│   └── transform/                 Core geospatial transformations
└── utils/
    ├── storage.py                 Path abstraction (S3 vs. local) + s3fs/boto3 helpers
    └── save_utils.py              S3 upload helpers

tests/
├── conftest.py                    Session-scoped moto S3 fixture
├── fixtures/                      Committed parquet/GeoJSON fixture files
├── generate_fixtures.py           One-time fixture generation (requires S3)
├── unit/                          Function-level unit tests
├── integration/                   Module-level integration tests
└── smoke/                         Full end-to-end smoke test (skipped in CI)

.github/workflows/test.yml         CI: pytest on PRs to dev
pyproject.toml                     uv / hatchling build configuration
```

## Data sources and acknowledgements

A comprehensive table of citations for data used in this analysis can be found in
[asf_heat_pump_suitability/config/README.md](asf_heat_pump_suitability/config/README.md).

- Contains OS data © Crown copyright and database right 2025.
- Contains Royal Mail data © Royal Mail copyright and database right 2025.
- Contains Office for National Statistics information licensed under the Open Government Licence v.3.0.
- Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- This information is subject to Crown copyright and database rights 2025 and is reproduced with the
  permission of HM Land Registry. See INSPIRE index polygons
  [conditions of use](https://use-land-property-data.service.gov.uk/datasets/inspire#conditions).
- The polygons (including the associated geometry, namely x, y co-ordinates) are subject to Crown
  copyright and database rights 2025 Ordnance Survey 100026316.
- Microsoft GlobalMLBuildingFootprints are made available under the
  [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/).
  Any rights in individual contents of the database are licensed under the
  [Database Contents License](http://opendatacommons.org/licenses/dbcl/1.0/).
- This work uses designated Historic Asset GIS Data, The Welsh Historic Environment Service (Cadw),
  2025, licensed under the Open Government Licence.
- This work uses Historic England data © Historic England 2025. Contains Ordnance Survey data
  © Crown copyright and database right 2025.
- This work uses data provided by the Consumer Data Research Centre, an ESRC Data Investment.
- This work uses Registers of Scotland's land extent polygons.
  © Crown copyright. Reproduced with the permission of Registers of Scotland.

## License

This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>.</small>
