# Local Development Setup

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
git clone https://github.com/nestauk/asf_heat_pump_suitability.git
cd asf_heat_pump_suitability
uv sync --group dev
```

## Running Tests

All tests use [moto](https://github.com/getmoto/moto) to mock S3 — no real AWS credentials are required:

```bash
uv run pytest tests/
```

With coverage:

```bash
uv run pytest tests/ --cov=asf_heat_pump_suitability --cov-report=term-missing
```

## Generating Fixture Files

The committed fixture files in `tests/fixtures/` were generated from real S3 data. To regenerate them (requires S3 access):

```bash
uv run python tests/generate_fixtures.py
```

Commit the outputs to git so CI can use them without S3 access.

## Code Quality

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Lint + fix
uv run ruff check . --fix
```

## Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
