# Refactor Specification

## Overview

Refactor the entire repository to eliminate dead code, modernise the toolchain, and
establish a well-tested, cloud-runnable pipeline. The pipeline takes UPRNs with
geospatial coordinates as input and outputs UPRNs with features required for the
heat pump suitability decision tree.

---

## Goals

- Cut out all excess code, files, and dependencies
- Switch to **uv** and **Python 3.12** (3.13 deferred until geospatial stack stabilises)
- Restructure the pipeline so each step is an independent CLI script
- Support running the full pipeline on the cloud via **arm_orbit**
- Support running locally with mocked S3 (moto) — no real data written to S3
- Run tests automatically on PRs to `dev` via GitHub Actions (credential-free)
- Comprehensive test coverage: function-level unit tests, module-level integration
  tests, and inter-stage schema contract tests
- Merge the unmerged `train_model.py` PR and include the flat classifier in the pipeline
- Replace Sphinx with mkdocs consistent with ds-cookiecutter standard

---

## What to Keep

Keep exactly the code that is **directly referenced in the pipeline**, plus anything
it imports transitively:

- `pipeline/run/` — main pipeline entry modules
- `pipeline/transform/` — UPRN filtering and transformation
- `pipeline/impute/` — flat property imputation
- `pipeline/prepare_features/` — feature preparation helpers
- `asf_heat_pump_suitability/getters/` — data loaders (S3, URLs, etc.)
- `asf_heat_pump_suitability/utils/` — geospatial and other utilities
- `config/` — YAML config (trimmed of dead keys)
- Companion scripts: `run_stream_inspire_files.py`, `train_model.py` (after merge)

**Hard-delete everything else**, including:

- `pipeline/flows/` (Metaflow — dropped entirely)
- `pipeline/run_scripts/`
- `pipeline/reweight_epc/`
- `pipeline/suitability/`
- `pipeline/sampling/`
- `pipeline/evaluation/`
- `pipeline/data_qc/`
- `setup.py`, `setup.cfg`, `requirements.txt`, `requirements_dev.txt`,
  `environment.yaml`, `Makefile`, `MANIFEST.in`, `jupytext.toml`
- All Jupyter notebooks and `notebooks/` directory
- `docs/` (to be replaced with mkdocs)

---

## Toolchain

### Package Manager

Switch from pip/conda to **uv**.

- Single `pyproject.toml` at the repo root.
- The package `asf_heat_pump_suitability` is installable: `uv pip install -e .`
- arm_orbit auto-detects `pyproject.toml` and installs via `uv pip install .` when
  launching cloud jobs.

### pyproject.toml Structure

Follow the ds-cookiecutter template (`{{ cookiecutter.module_name }}/pyproject.toml`)
exactly:

- Build backend: **hatchling** (`requires = ["hatchling"]`)
- Version sourced from `asf_heat_pump_suitability/__init__.py`
- Python constraint: `>=3.12`
- Ruff for both linting and formatting (replacing black + flake8 + isort)
- Ruff configuration copied verbatim from the ds-cookiecutter template:
  - Line length: 120
  - Rules: ANN, B, C, E, F, I, N, W
  - Ignored: D100, S101, ANN002, ANN003
  - Per-file ignores for `__init__.py`: F401, E402, D104
  - Docstring convention: Google
  - `known-first-party = ["asf_heat_pump_suitability"]`
- Dependency groups: `dev = ["ipykernel", "jupytext", "nbstripout", "ruff", "pytest", "pre-commit", "moto[s3]", "pytest-cov"]`

### Linting and Formatting

Use **ruff** (both lint and format). Remove all references to black, flake8, isort,
and setup.cfg.

### Pre-commit Hooks

Use the ds-cookiecutter `.pre-commit-config.yaml` template verbatim (no Jinja
variables; this is a Python-only repo, `use_r = no`):

```yaml
repos:
  - repo: https://gitlab.com/vojko.pribudic.foss/pre-commit-update
    rev: v0.9.0
    hooks:
      - id: pre-commit-update
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: end-of-file-fixer
        exclude: ^.cookiecutter/config
      - id: trailing-whitespace
      - id: check-toml
      - id: check-yaml
        exclude: "docs/mkdocs.yml"
      - id: no-commit-to-branch
        args: ["-b", dev, "-b", main]
        pass_filenames: false
  - repo: https://github.com/nestauk/pre-commit-hooks
    rev: v1.2.0
    hooks:
      - id: nbstripout-preserve-timestamp
      - id: jupytext-enforce-pairing
      - id: jupytext-smart-sync
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.9
    hooks:
      - id: ruff-format
      - id: ruff-check
  - repo: https://github.com/prettier/pre-commit
    rev: v2.1.2
    hooks:
      - id: prettier
```

### System Dependencies (GDAL)

Do **not** use conda for GDAL. Use modern PyPI wheel packages for all geospatial
libraries. geopandas ≥1.0, shapely ≥2.1, pyproj ≥3.7, and fiona ship bundled GDAL
wheels on Python 3.12. No system install or conda required.

---

## Pipeline Structure

### Entry Points (Flat Layout)

Each main pipeline step is a standalone CLI script at `pipeline/{step_name}.py`.
Supporting modules stay in their existing sub-packages.

```
pipeline/
├── uprns.py            # Step 1: Filter OS UPRNs to domestic
├── add_features.py     # Step 2: Add features to domestic UPRNs
├── setup/              # Companion scripts (run once, not every pipeline run)
│   ├── train_model.py      # Train flat classifier (merged from unmerged PR)
│   └── stream_inspire_files.py  # Stream land registry data to S3
├── impute/             # Imputation modules
├── prepare_features/   # Feature preparation modules
├── transform/          # UPRN filtering and transformation modules
└── run/                # (legacy support — may be consolidated into top-level scripts)
```

### Step Sequencing

Steps are independent CLIs. They communicate via **shared config-defined S3 paths**
(convention over configuration). No explicit `--input-path` argument is passed
between steps; each step reads its input and writes its output to the paths defined
in the Pydantic config.

To run the full pipeline:

```bash
python pipeline/uprns.py [--area <area>]
python pipeline/add_features.py
```

To run on the cloud via arm_orbit:

```bash
orbit launch --script pipeline/uprns.py --team <team> --project <project>
orbit launch --script pipeline/add_features.py --team <team> --project <project>
```

(arm_orbit handles packaging, Docker build, and Batch submission automatically.)

### Companion Scripts

Run once before the main pipeline (or whenever source data changes):

```bash
python pipeline/setup/stream_inspire_files.py -n all   # Ingest land registry
python pipeline/setup/train_model.py                   # Train flat classifier → S3
```

### Geographic Filtering

`uprns.py` preserves the existing `--area` flag with the following modes:

- `plymouth` (default for dev/test)
- `plymouth_similar`
- `sampling`
- `gb` (full Great Britain)

Region grid square lists move from hardcoded to the Pydantic config model
(see Config section).

---

## Configuration

### Config System

Replace the free-form `config/base.yaml` with a **Pydantic-settings** model.
Config loads from YAML as a base, with environment variable overrides for any field.

```
config/
├── base.yaml       # Trimmed YAML — only keys still used by active code
└── settings.py     # Pydantic BaseSettings models that load and validate the YAML
```

`settings.py` defines typed models covering:

- S3 paths for inputs and pipeline outputs
- Grid square codes per area (plymouth, plymouth_similar, sampling, gb)
- Feature mappings (property type, building year, RUC, etc.)
- Model artifact paths (flat classifier S3 path)
- Data source URLs (INSPIRE, EPC, etc.)

Config is instantiated once at pipeline startup and passed through explicitly.

### Environment Variable Overrides

Any Pydantic field can be overridden by a correspondingly-named environment variable.
Key env vars:

| Variable           | Purpose                                     | Default                     |
| ------------------ | ------------------------------------------- | --------------------------- |
| `DATA_MODE`        | `s3` or `local`                             | `s3`                        |
| `AWS_ENDPOINT_URL` | Override S3 endpoint (set by moto in tests) | unset                       |
| `S3_BUCKET`        | Override default S3 bucket                  | `asf-heat-pump-suitability` |
| `ORBIT_ENV`        | arm_orbit environment (dev/staging/prod)    | `prod`                      |

---

## Storage Abstraction

A thin **path-only abstraction** in `asf_heat_pump_suitability/utils/storage.py`:

```python
def get_path(key: str, config: Settings) -> str:
    """Return the appropriate path for a data key.

    When DATA_MODE=local, returns a local filesystem path under outputs/.
    When DATA_MODE=s3 (default), returns the configured S3 URI.
    """
```

All pipeline S3 I/O uses this function to obtain paths. The actual reads and writes
remain as-is (polars, geopandas, s3fs) — they just receive the correct path string.
When `DATA_MODE=local` is set, paths point to `outputs/` on the local filesystem
instead of S3.

For tests, moto patches boto3/s3fs at the `AWS_ENDPOINT_URL` level so pipeline code
is unmodified; the mock is transparent.

---

## S3 Paths

Keep the **existing S3 bucket and path structure** (`asf-heat-pump-suitability`).
No downstream consumers or dashboards should be broken. All paths are defined in
`config/base.yaml` and loaded via `Settings`.

The trained flat classifier model is stored at a config-defined S3 path:

- `train_model.py` saves the model artifact to S3
- `add_features.py` loads it from the same path at runtime

---

## Testing

### Strategy

All three levels of testing:

1. **Function-level unit tests** — test each function in `/transform`, `/impute`,
   `/prepare_features` with minimal (5–10 row) GeoDataFrames
2. **Module-level integration tests** — test the public interface of each pipeline
   module (e.g. `filter_gdf_residential_uprns`) with a small but realistic fixture
3. **Schema contract tests** — verify that stage 1 output matches stage 2 expected
   input schema; that column names, dtypes, and geometry types are correct at each
   boundary

Priority failures to catch:

- Schema drift (external data source changes column names/types mid-pipeline)
- Regression in feature calculations after refactoring

### Test Fixtures

**Small committed fixture files** checked into `tests/fixtures/`:

- Generated once by a `tests/generate_fixtures.py` script that samples real S3 data
  (requires S3 access; run locally by developers; output committed to git)
- Fixtures represent a realistic but tiny slice of the real pipeline data
  (e.g. ~100 domestic UPRNs from Plymouth, corresponding EPC records, land parcels,
  building footprints)
- Used for both unit tests (subsets) and the smoke test (full fixture set)

### Moto Setup

`tests/conftest.py` has a **session-scoped** pytest fixture that:

1. Starts a moto S3 mock context
2. Creates the `asf-heat-pump-suitability` bucket
3. Uploads all committed fixture files to their expected S3 paths
4. Yields (all tests in the session share this mock)
5. Tears down the mock

All pipeline code runs against this mock transparently via `AWS_ENDPOINT_URL`.

### CI Smoke Test

The CI runs the full pipeline on fixture data end-to-end:

```
uprns.py (fixture UPRNs) → add_features.py (fixture auxiliary data) → output parquet
```

The smoke test asserts:

- Pipeline completes without error
- Output has correct schema (columns, dtypes)
- Output row count is within expected range for the fixture input
- No UPRNs in the output that weren't in the input
- Key feature columns are non-null for expected records

### Test Organisation

```
tests/
├── conftest.py             # Session-scoped moto fixture, shared helpers
├── generate_fixtures.py    # One-time fixture generation script (not run in CI)
├── fixtures/               # Committed fixture parquet/geojson files
│   ├── uprns.parquet
│   ├── epc_domestic.parquet
│   ├── land_registry.gpkg
│   └── ...
├── unit/
│   ├── test_transform_uprns.py
│   ├── test_impute_property_type.py
│   ├── test_prepare_features_*.py
│   └── ...
├── integration/
│   ├── test_uprns_pipeline.py
│   └── test_add_features_pipeline.py
└── smoke/
    └── test_full_pipeline.py
```

---

## CI/CD (GitHub Actions)

### Trigger

PRs to `dev` only.

### Workflow

```yaml
on:
  pull_request:
    branches: [dev]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group dev
      - run: uv run pytest tests/ --cov=asf_heat_pump_suitability --cov-report=xml
```

No AWS credentials in CI. All S3 operations use moto.

---

## Documentation

Replace Sphinx with **mkdocs-material** (consistent with ds-cookiecutter).

- Delete `docs/` (Sphinx)
- Create `docs/` (mkdocs) with:
  - `mkdocs.yml` at repo root
  - `docs/index.md` — project overview
  - `docs/pipeline.md` — pipeline description, step-by-step run instructions
  - `docs/setup.md` — local dev setup (uv, fixture generation, running tests)
  - `docs/cloud.md` — running on the cloud via arm_orbit
  - `docs/config.md` — configuration reference

---

## Migration Steps (Suggested Order)

1. **Create new branch** from `dev` for the refactor
2. **Merge train_model.py PR** into the refactor branch
3. **Delete all dead files/directories** (hard-delete, no archiving)
4. **Replace setup.py/setup.cfg/requirements.txt/environment.yaml** with `pyproject.toml`
   (hatchling, Python ≥3.12, ruff, uv)
5. **Restructure pipeline**: move entry scripts to `pipeline/uprns.py`,
   `pipeline/add_features.py`; move companion scripts to `pipeline/setup/`
6. **Implement `config/settings.py`** (Pydantic-settings); trim `config/base.yaml`
7. **Implement storage abstraction** (`utils/storage.py`)
8. **Update all imports** throughout remaining codebase to reflect new structure
9. **Create test fixtures** (`tests/generate_fixtures.py`, run locally, commit output)
10. **Write tests** (unit → integration → smoke)
11. **Add GitHub Actions workflow** (`.github/workflows/test.yml`)
12. **Replace pre-commit config** with ds-cookiecutter version
13. **Set up mkdocs** and write docs
14. **Open PR** to `dev`
