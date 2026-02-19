"""Smoke test: run the full pipeline end-to-end on fixture data.

Verifies that:
- The pipeline completes without error
- Output has the expected schema (columns, non-null key fields)
- Output row count is within the expected range for the fixture input
- No UPRNs in the output that weren't in the input
"""

import pytest


@pytest.mark.skip(reason="Requires committed fixture files — run after generate_fixtures.py")
def test_full_pipeline_smoke(tmp_path: pytest.TempdirFactory, s3_bucket: None) -> None:  # noqa: ANN001
    """Full pipeline smoke test: uprns → add_features → output parquet.

    Steps:
    1. Run pipeline/uprns.py on fixture UPRNs (mocked S3)
    2. Run pipeline/add_features.py on the uprns output
    3. Assert output schema and basic data quality

    This test is skipped in CI until fixture files are committed.
    Remove the @pytest.mark.skip decorator after running generate_fixtures.py.
    """
    import polars as pl

    from asf_heat_pump_suitability import config
    from pipeline.add_features import run as add_features_run
    from pipeline.uprns import run as uprns_run

    # Step 1: Filter UPRNs (using committed fixture data in mocked S3)
    uprns_run(area="plymouth")

    uprns_output_path = config["output"]["residential_uprns_template"].format(area="plymouth")

    # Verify uprns output
    uprns_df = pl.read_parquet(uprns_output_path)
    assert len(uprns_df) > 0, "Expected at least some residential UPRNs"
    assert "UPRN" in uprns_df.columns
    assert "X_COORDINATE" in uprns_df.columns
    assert "Y_COORDINATE" in uprns_df.columns
    assert uprns_df["UPRN"].null_count() == 0

    # Step 2: Add features
    add_features_run(uprns_path=uprns_output_path)

    features_output_path = config["output"]["features_template"].format(uprns_stem="plymouth_residential_uprns")
    features_df = pl.read_parquet(features_output_path)

    # Verify features output
    assert len(features_df) > 0
    assert "UPRN" in features_df.columns
    assert "property_type_flat" in features_df.columns

    # No UPRNs in output that weren't in input
    input_uprns = set(uprns_df["UPRN"].to_list())
    output_uprns = set(features_df["UPRN"].to_list())
    assert output_uprns.issubset(input_uprns), "Output contains UPRNs not in input"

    # Row count should be within expected range (some UPRNs may be dropped)
    assert len(features_df) <= len(uprns_df)
    assert len(features_df) >= int(len(uprns_df) * 0.5), "More than 50% of UPRNs were dropped"
