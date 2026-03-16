"""CLI: Filter OS UPRNs to domestic-only and save per-LAD outputs.

Filters all UK UPRNs to residential UPRNs only. A UPRN is assumed to be residential if it:
- Is geolocated inside a building footprint AND not in a non-residential building type AND not in the
  non-domestic EPC register, OR
- Appears in the domestic EPC register.

Outputs are written to ``{output_dir}/{lad-slug}/domestic_uprns.parquet`` for each LAD.

Example usage:

    # Full GB run (writes to S3)
    LOCAL_DEV=false uv run python asf_heat_pump_suitability/pipeline/uprns.py

    # Single LAD by slug
    uv run python asf_heat_pump_suitability/pipeline/uprns.py --lad manchester

    # Multiple LADs
    uv run python asf_heat_pump_suitability/pipeline/uprns.py --lad manchester --lad salford

    # Custom output directory
    uv run python asf_heat_pump_suitability/pipeline/uprns.py --lad manchester --output-dir /data/outputs/
"""

from typing import Optional

import polars as pl
import typer

from asf_heat_pump_suitability.config.settings import Settings
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata, load_tree_input
from asf_heat_pump_suitability.getters.geography_lookup import get_geography_lookup, resolve_lads
from asf_heat_pump_suitability.pipeline.transform import non_residential_entities, poi, uprns
from asf_heat_pump_suitability.utils import save_utils

app = typer.Typer(help=__doc__)


@app.command()
def main(
    lad: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--lad",
        help=(
            "LAD name, slug, or ONS code. May be repeated to process multiple LADs. " "Defaults to all LADs if not set."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for this run only.",
    ),
) -> None:
    """Filter OS UPRNs to domestic-only and save per-LAD results as parquet."""
    settings = Settings(output_dir=output_dir) if output_dir else Settings()

    # Resolve LAD codes
    lookup = get_geography_lookup()
    lad_codes = resolve_lads(lads=lad, utlas=None, combineds=None, lookup=lookup)
    print(f"Processing {len(lad_codes)} LAD(s).")

    # Load data shared across all LADs
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    poi_gdf = load_tree_input.load_gdf_poi()
    poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=poi.load_set_non_domestic_poi_categories(),
    )

    all_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries()

    # Process each LAD
    for lad_code in lad_codes:
        row = lookup[lookup["lad_code"] == lad_code].iloc[0]
        lad_name = row["lad_name"]
        lad_slug = row["lad_slug"]
        print(f"\n--- Processing LAD: {lad_name} ({lad_code}) ---")

        lad_boundary = all_boundaries_gdf[all_boundaries_gdf["LAD23CD"] == lad_code][["LAD23CD", "LAD23NM", "geometry"]]

        # Filter UPRNs to this LAD
        lad_uprns_gdf = uprns_gdf.sjoin(
            lad_boundary,
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

        # Load buildings for this LAD using pyogrio mask
        layers = {
            f"{layer}_gdf": load_tree_input.load_openmap_local_layer(layer=layer, lad_boundary=lad_boundary)
            for layer in ["important_building", "railway_station", "building"]
        }

        non_residential_buildings_gdf = non_residential_entities.generate_gdf_non_residential_buildings(
            **layers, poi_gdf=poi_gdf, uprns_gdf=lad_uprns_gdf
        )

        residential_uprns_gdf = uprns.filter_gdf_residential_uprns(
            uprn_gdf=lad_uprns_gdf,
            buildings_gdf=layers["building_gdf"],
            non_residential_buildings_gdf=non_residential_buildings_gdf,
        )

        output_cols = ["UPRN", "X_COORDINATE", "Y_COORDINATE", "LATITUDE", "LONGITUDE", "LAD23CD", "LAD23NM"]
        df = pl.from_pandas(residential_uprns_gdf[[c for c in output_cols if c in residential_uprns_gdf.columns]])

        print(f"Saving {len(df):,} domestic UPRNs for {lad_name}...")
        out_path = settings.resolve_output_path(lad_slug, "domestic_uprns.parquet")
        save_utils.save_df(df, out_path)

    print("\nDone.")


if __name__ == "__main__":
    app()
