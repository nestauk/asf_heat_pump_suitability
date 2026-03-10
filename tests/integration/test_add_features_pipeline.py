"""Integration tests for pipeline/add_features.py.

These tests run the full add_features pipeline step end-to-end using committed
fixture files. S3 writes are mocked with moto; external S3 reads are mocked with
monkeypatch so no real AWS credentials are required.

Run to generate fixture files first:
    python tests/generate_fixtures.py
"""

import pytest


@pytest.mark.skip(reason="fixtures not generated — run tests/generate_fixtures.py first")
def test_add_features_pipeline_produces_parquet(tmp_path, monkeypatch):
    """Full add_features.py step produces a valid parquet file."""
    # This test will be enabled once fixture files are committed.
    # Expected behaviour:
    # 1. Load fixture domestic_uprns.parquet from tests/fixtures/
    # 2. Run add_features logic (mock S3 reads)
    # 3. Assert output parquet exists and has expected feature columns
    pass


@pytest.mark.skip(reason="fixtures not generated — run tests/generate_fixtures.py first")
def test_add_features_pipeline_output_schema(tmp_path, monkeypatch):
    """Output parquet from add_features.py has the expected schema."""
    # Expected columns include:
    # UPRN, X_COORDINATE, Y_COORDINATE, property_type_flat, in_block_of_flats,
    # max_contiguous_outdoor_space_area_m2, total_outdoor_space_area_m2
    pass
