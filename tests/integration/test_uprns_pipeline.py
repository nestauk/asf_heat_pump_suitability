"""Integration tests for pipeline/uprns.py.

These tests run the full uprns pipeline step end-to-end using committed fixture
files. S3 writes are mocked with moto; external S3 reads are mocked with
monkeypatch so no real AWS credentials are required.

Run to generate fixture files first:
    python tests/generate_fixtures.py
"""

import pytest


@pytest.mark.skip(reason="fixtures not generated — run tests/generate_fixtures.py first")
def test_uprns_pipeline_plymouth_produces_parquet(tmp_path, monkeypatch):
    """Full uprns.py step produces a valid parquet file for Plymouth."""
    # This test will be enabled once fixture files are committed.
    # Expected behaviour:
    # 1. Load fixture UPRNs from tests/fixtures/
    # 2. Run filter logic
    # 3. Assert output parquet exists and has expected columns
    pass


@pytest.mark.skip(reason="fixtures not generated — run tests/generate_fixtures.py first")
def test_uprns_pipeline_output_schema(tmp_path, monkeypatch):
    """Output parquet from uprns.py has the expected schema."""
    # Expected columns: UPRN, X_COORDINATE, Y_COORDINATE, LATITUDE, LONGITUDE,
    # optionally LAD23CD, LAD23NM
    pass
