# ASF Heat Pump Suitability

Generates property-level data for Nesta's local heat planning tool, which clusters
residential properties by the most suitable low-carbon heating technology (air source heat
pump, ground source heat pump, shared ground loop, district heat network, or communal heat
source) and provides supporting information about each cluster.

---

## Pipeline overview

```
OS Open UPRN  ──┐
EPC register  ──┼──▶  uprns.py  ──▶  domestic_uprns.parquet
Building       ─┘               (filter to domestic only)
footprints
                                        │
         ┌──────────────────────────────┘
         ▼
Land registry ─┐
Building       ─┼──▶  add_features.py  ──▶  uprns_with_features.parquet
footprints     ─┘    (add property type,       (ready for decision tree)
Block-of-flats        block-of-flats flag,
classifier            outdoor space)
```

| Script | Input | Output | Description |
|---|---|---|---|
| `pipeline/uprns.py` | S3: OS Open UPRN, EPC, building footprints | `domestic_uprns.parquet` | Filter all UK UPRNs to domestic-only |
| `pipeline/add_features.py` | `domestic_uprns.parquet` + S3 data | `uprns_with_features.parquet` | Add features for decision tree |
| `pipeline/run.py` | — | — | Orchestrate both steps in order |

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [direnv](https://direnv.net/) (optional, for automatic env vars)
- AWS credentials with read access to `s3://asf-heat-pump-suitability` (for pipeline runs)

### Install

```bash
git clone <repo-url>
cd asf_heat_pump_suitability
uv sync --all-extras
```

### Environment variables

The pipeline reads two environment variables:

| Variable | Default | Description |
|---|---|---|
| `LOCAL_DEV` | `true` | When `true`, pipeline outputs are written to the local filesystem. Set `false` for production/cloud runs that write to S3. |
| `OUTPUT_DIR` | `./outputs/` (local) or `s3://asf-heat-pump-suitability/outputs/` (cloud) | Base directory for all pipeline outputs. |

**Using direnv** (recommended): run `direnv allow` to automatically set `LOCAL_DEV=true` and
`OUTPUT_DIR=./outputs/` whenever you enter the repo directory. This prevents accidental writes
to S3 from a developer laptop.

**Manually**: export the variables before running any pipeline script:

```bash
export LOCAL_DEV=true
export OUTPUT_DIR=./outputs/
```

---

## Running the pipeline

### Full pipeline (both steps)

```bash
# Local run — Plymouth only, writes to ./outputs/
uv run python pipeline/run.py --local-authorities plymouth

# Greater Manchester
uv run python pipeline/run.py --local-authorities greater_manchester_las

# Production run — full GB, writes to S3
LOCAL_DEV=false uv run python pipeline/run.py
```

### Individual steps

```bash
# Step 1: filter UPRNs to domestic
uv run python pipeline/uprns.py --local-authorities plymouth

# Step 2: add features (reads from OUTPUT_DIR/domestic_uprns.parquet by default)
uv run python pipeline/add_features.py --local-authorities plymouth

# Step 2 with explicit input path
uv run python pipeline/add_features.py \
    --uprns ./outputs/domestic_uprns.parquet \
    --local-authorities plymouth
```

### Available `--local-authorities` presets

| Preset | Coverage |
|---|---|
| `plymouth` | Plymouth |
| `plymouth_similar_cities` | Plymouth, Liverpool, Portsmouth, Southampton, Swansea |
| `sampling_areas` | Bath, Bradford, Glasgow, Manchester, Nottingham, Plymouth |
| `greater_manchester_las` | All 10 Greater Manchester local authorities |

Omit `--local-authorities` to process all of GB (not yet fully scaled).

---

## Companion scripts

These scripts are run once to set up data that the pipeline depends on.

### `pipeline/setup/train_model.py` — train the block-of-flats classifier

Trains the Random Forest classifier that `add_features.py` uses to predict whether a
building is a block of flats. Run this once when new labelled training data is available.

```bash
uv run python pipeline/setup/train_model.py \
    --uprns s3://asf-heat-pump-suitability/local_heat_planning/outputs/sampling_areas_residential_uprns.parquet \
    --labelled-data s3://asf-heat-pump-suitability/local_heat_planning/inputs/processed/manually_labelled_block_of_flats.parquet \
    --save
```

### `pipeline/setup/stream_inspire_files.py` — stream INSPIRE land registry files to S3

Downloads INSPIRE land parcel polygons from the HMLR (England & Wales) and Registers of
Scotland websites and streams them to S3. Run this quarterly when INSPIRE data is updated.

```bash
# Stream both England & Wales and Scotland
uv run python pipeline/setup/stream_inspire_files.py --nations all

# England & Wales only
uv run python pipeline/setup/stream_inspire_files.py --nations ew
```

---

## Running tests

```bash
uv run pytest                          # all tests
uv run pytest tests/unit/              # unit tests only
uv run pytest tests/integration/       # integration tests (some skipped until fixtures generated)
uv run pytest --cov=asf_heat_pump_suitability  # with coverage
```

### Generating test fixtures

Integration tests require small fixture files committed to `tests/fixtures/`. Generate
them once with real S3 access, then commit the results:

```bash
python tests/generate_fixtures.py
git add tests/fixtures/
```

---

## Development commands

```bash
uv sync --all-extras              # install all deps including dev
uv run pytest                    # run tests
uv run ruff check .              # lint
uv run ruff format .             # format
uv run ruff check --fix .        # lint and auto-fix
uv run python pipeline/uprns.py --help  # CLI help
```

---

## Contributing

### Function naming convention

All functions follow `action_returntype_description()`:

- **action**: `load`, `transform`, `generate`, `extend`, `filter`, `predict`, `impute`
- **returntype**: `df` (Polars), `gdf` (GeoDataFrame), `dict`, `set`, `list`
- **description**: short description of the function, e.g. `osopen_uprn`, `residential_uprns`

Examples: `load_df_osopen_uprn()`, `filter_gdf_residential_uprns()`, `generate_gdf_uprn_coords()`

### Docstrings

All public functions use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
with `Args` and `Returns` sections including types.

### S3 file naming convention

Raw source data in `s3://asf-heat-pump-suitability/` follows `a_vb_c_d_e`:

- **a**: Publication date
- **b**: Version date (if available)
- **c**: Source name (e.g. `OSOpen`, `ONS`)
- **d**: Short descriptive name
- **e**: Geographical coverage

Example: `v202510_OSOpenMapLocal_building_geometries_Plymouth.shp`

Record all data sources in `asf_heat_pump_suitability/config/README.md`.

---

## Data sources

A comprehensive table of data sources, citations, and attributions is in
[asf_heat_pump_suitability/config/README.md](asf_heat_pump_suitability/config/README.md).

Key sources:

- **UPRNs**: OS Open UPRN (Crown copyright)
- **EPC**: Domestic EPC register via `asf-daps` lakehouse; Commercial EPC (HMLR)
- **Building footprints**: Microsoft Global ML Building Footprints (ODbL)
- **Boundaries**: ONS LAD boundaries (OGL v3)
- **Land parcels**: INSPIRE Index Polygons (HMLR / Crown copyright)
- **Grid capacity**: ENW, Northern Powergrid, SPEN, SSEN, UKPN, NGED

### Attributions

- Contains OS data © Crown copyright and database right 2025.
- Contains Royal Mail data © Royal Mail copyright and database right 2025.
- Contains Office for National Statistics information licensed under the Open Government Licence v.3.0.
- This information is subject to Crown copyright and database rights 2025 and is reproduced with the permission of HM Land Registry.
- Microsoft GlobalMLBuildingFootprints are made available under the [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/).
- This work uses designated Historic Asset GIS Data, The Welsh Historic Environment Service (Cadw), 2025, licensed under the Open Government Licence.
- This work uses Historic England data © Historic England 2025.
- This work uses data provided by the Consumer Data Research Centre, an ESRC Data Investment.
- This work uses Registers of Scotland's land extent polygons. © Crown copyright.
- This work uses data from the Scottish Census © Crown copyright. Data supplied by National Records of Scotland.
- This work uses data from Electricity North West Ltd, SP Energy Networks, SSEN Distribution, UK Power Networks. [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Supported by Northern Powergrid Open Data. [License](https://northernpowergrid.opendatasoft.com/p/opendatalicence/)
- Supported by NGED Open Data. [License](https://www.nationalgrid.co.uk/open-data-licence)

---

## License

This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License.

<small>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>.</small>
