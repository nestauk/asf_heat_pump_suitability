"""Script to (re)generate committed fixture files for integration tests.

**When to run**: Run this script once when data sources change (e.g. a new UPRN
snapshot is published) to refresh the committed fixture files in tests/fixtures/.

**Requires**: Real AWS credentials with read access to S3 bucket asf-heat-pump-suitability.

Usage:
    python tests/generate_fixtures.py

Fixture files generated:
    tests/fixtures/uprns.parquet       — small slice of OS Open UPRN data (Plymouth)
    tests/fixtures/buildings.parquet   — building footprints for Plymouth grid square SX
"""

from pathlib import Path

import polars as pl

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def generate_uprns_fixture() -> None:
    """Pull a small slice of domestic UPRNs from S3 and save to tests/fixtures/.

    Selects 200 UPRNs from the Plymouth area (grid square SX) as a representative
    sample for integration testing.
    """
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import load_boundaries, load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns

    print("Loading OS Open UPRN data from S3...")
    df = load_geodata.load_df_osopen_uprn()
    gdf = uprns.generate_gdf_uprn_coords(df)

    # Filter to Plymouth for a small, representative sample
    la_boundaries = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=config["constant"]["plymouth"]["la_names"]
    )
    gdf = gdf.sjoin(
        la_boundaries[["geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns="index_right")

    # Keep a small slice
    sample = pl.from_pandas(gdf[["UPRN", "X_COORDINATE", "Y_COORDINATE"]]).head(200)

    out_path = FIXTURES_DIR / "uprns.parquet"
    sample.write_parquet(out_path)
    print(f"Saved {len(sample)} UPRNs to {out_path}")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(exist_ok=True)
    generate_uprns_fixture()
    print("All fixtures generated.")
