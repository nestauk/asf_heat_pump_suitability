"""Setup script: build the ONS geography lookup and upload to S3.

Fetches the LAD->UTLA (England & Wales) and LAD->Combined Authority (England)
mappings from the ONS ArcGIS API, adds Scottish council areas from the LAD
boundary file, and writes the combined lookup as parquet to S3.

Run once (or whenever ONS source data is updated):

    python pipeline/setup/build_geography_lookup.py
"""

import geopandas as gpd
import pandas as pd
import requests
import s3fs
import typer

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters.geography_lookup import make_slug

app = typer.Typer(help=__doc__)

_MAX_RECORDS = 2000  # ONS datasets are ~300-400 rows; this is a safe upper bound


def _fetch_ons_feature_service(base_url: str) -> pd.DataFrame:
    """Fetch all records from an ONS ArcGIS FeatureServer layer.

    Appends the standard ArcGIS query parameters to ``base_url`` and returns
    the attribute table as a DataFrame.

    Args:
        base_url: FeatureServer layer URL (no trailing slash, no /query suffix).

    Returns:
        DataFrame of all feature attribute records.
    """
    url = f"{base_url}/query" f"?outFields=*&where=1%3D1&f=geojson&resultRecordCount={_MAX_RECORDS}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    features = resp.json()["features"]
    return pd.DataFrame([f["properties"] for f in features])


@app.command()
def main() -> None:
    """Fetch ONS geography data, merge with Scottish LADs, and upload lookup to S3."""
    # 1. Fetch England & Wales LAD -> UTLA
    print("Fetching LAD->UTLA lookup (England & Wales)...")
    lad_utla_url = config["inputs"]["reference"]["lad_utla_lookup"]
    lad_utla_df = _fetch_ons_feature_service(lad_utla_url)
    print(f"  {len(lad_utla_df)} rows")

    # Normalise to expected column names
    lad_utla_df = lad_utla_df.rename(
        columns={
            "LAD23CD": "lad_code",
            "LAD23NM": "lad_name",
            "UTLA23CD": "utla_code",
            "UTLA23NM": "utla_name",
        }
    )[["lad_code", "lad_name", "utla_code", "utla_name"]]

    # 2. Fetch England LAD -> Combined Authority
    print("Fetching LAD->Combined Authority lookup (England)...")
    lad_combined_url = config["inputs"]["reference"]["lad_combined_lookup"]
    lad_combined_df = _fetch_ons_feature_service(lad_combined_url)
    print(f"  {len(lad_combined_df)} rows")

    lad_combined_df = lad_combined_df.rename(
        columns={
            "LAD23CD": "lad_code",
            "CAUTH23CD": "combined_code",
            "CAUTH23NM": "combined_name",
        }
    )[["lad_code", "combined_code", "combined_name"]]

    # 3. Read Scottish council areas from the LAD boundary file
    print("Reading Scottish LADs from boundary file...")
    boundaries_gdf = gpd.read_file(config["inputs"]["geodata"]["lad_boundaries"])
    scottish_df = (
        boundaries_gdf[boundaries_gdf["LAD23CD"].str.startswith("S")][["LAD23CD", "LAD23NM"]]
        .rename(columns={"LAD23CD": "lad_code", "LAD23NM": "lad_name"})
        .assign(utla_code=lambda df: df["lad_code"], utla_name=lambda df: df["lad_name"])
        .copy()
    )
    print(f"  {len(scottish_df)} Scottish council areas")

    # 4. Merge England & Wales + Scotland
    ew_lookup = lad_utla_df.merge(lad_combined_df, on="lad_code", how="left")
    scottish_df["combined_code"] = None
    scottish_df["combined_name"] = None
    lookup = pd.concat([ew_lookup, scottish_df], ignore_index=True)

    # 5. Derive slugs for all three tiers
    lookup["lad_slug"] = lookup["lad_name"].map(make_slug)
    lookup["utla_slug"] = lookup["utla_name"].map(make_slug)
    lookup["combined_slug"] = lookup["combined_name"].map(lambda x: make_slug(x) if pd.notna(x) else None)

    # Enforce column order
    lookup = lookup[
        [
            "lad_code",
            "lad_name",
            "lad_slug",
            "utla_code",
            "utla_name",
            "utla_slug",
            "combined_code",
            "combined_name",
            "combined_slug",
        ]
    ]

    print(f"Lookup has {len(lookup)} LADs total.")

    # 6. Write to S3
    out_path = config["inputs"]["reference"]["geography_lookup"]
    print(f"Writing to: {out_path}")
    fs = s3fs.S3FileSystem()
    with fs.open(out_path, "wb") as f:
        lookup.to_parquet(f, index=False)

    print("Done.")


if __name__ == "__main__":
    app()
