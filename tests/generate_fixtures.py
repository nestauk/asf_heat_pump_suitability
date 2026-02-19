"""Generate small fixture files for the test suite.

This script samples real S3 data to produce tiny but realistic fixtures
committed to tests/fixtures/. It requires valid AWS S3 access and should
be run locally by developers; its outputs are committed to git and used
in CI without S3 access (via moto).

Run:
    python tests/generate_fixtures.py

Outputs (written to tests/fixtures/):
    uprns.parquet               ~100 domestic UPRNs from Plymouth
    epc_domestic.parquet        Matching EPC domestic records
"""

from pathlib import Path

import polars as pl

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

# Plymouth bounding box (BNG EPSG:27700)
PLYMOUTH_BBOX = (230000, 45000, 260000, 75000)
N_UPRN_SAMPLE = 100


def generate_uprn_fixture() -> None:
    """Sample ~100 Plymouth domestic UPRNs and write to fixtures/uprns.parquet."""
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import base_getters

    path = config["data"]["geodata"]["uk_osopen_uprn"]
    filename = path.split("/")[-1].replace("_csv.zip", "")
    df = base_getters.get_df_from_zip_csv_s3(path, extract_file=f"{filename}.csv")

    # Filter to Plymouth bounding box
    df = df.filter(
        (pl.col("X_COORDINATE") >= PLYMOUTH_BBOX[0])
        & (pl.col("X_COORDINATE") <= PLYMOUTH_BBOX[2])
        & (pl.col("Y_COORDINATE") >= PLYMOUTH_BBOX[1])
        & (pl.col("Y_COORDINATE") <= PLYMOUTH_BBOX[3])
    ).head(N_UPRN_SAMPLE)

    out = FIXTURES_DIR / "uprns.parquet"
    df.write_parquet(out)
    print(f"Wrote {len(df)} rows to {out}")


def generate_epc_fixture() -> None:
    """Sample EPC domestic records matching fixture UPRNs and write to fixtures/epc_domestic.parquet."""
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import base_getters

    uprns = pl.read_parquet(FIXTURES_DIR / "uprns.parquet")["UPRN"].to_list()

    path = config["data"]["epc"]["domestic"]
    df = base_getters.load_df_from_s3(path, columns=["UPRN", "CURRENT_ENERGY_RATING"])
    df = df.filter(pl.col("UPRN").is_in(uprns))

    out = FIXTURES_DIR / "epc_domestic.parquet"
    df.write_parquet(out)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    print("Generating fixture files (requires S3 access)...")
    generate_uprn_fixture()
    generate_epc_fixture()
    print("Done.")
