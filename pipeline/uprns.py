"""CLI: Filter OS UPRNs to domestic-only and save the result.

Filters all UK UPRNs to residential UPRNs only. A UPRN is assumed to be residential if it:
- Is geolocated inside a building footprint AND not in a non-residential building type AND not in the
  non-domestic EPC register, OR
- Appears in the domestic EPC register.

Example usage:

    # Full GB run (writes to S3)
    LOCAL_DEV=false uv run python pipeline/uprns.py

    # Plymouth only (writes to ./outputs/)
    uv run python pipeline/uprns.py --local-authorities plymouth

    # Greater Manchester only with custom output directory
    uv run python pipeline/uprns.py --local-authorities greater_manchester_las --output-dir /data/outputs/
"""

import os
from typing import Optional

import polars as pl
import typer

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.config.settings import Settings
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata, load_tree_input
from asf_heat_pump_suitability.pipeline.transform import non_residential_entities, poi, uprns
from asf_heat_pump_suitability.utils import save_utils

app = typer.Typer(help=__doc__)


@app.command()
def main(
    local_authorities: Optional[str] = typer.Option(
        None,
        "--local-authorities",
        help=(
            "Local authority preset or space-separated LA codes. "
            "Available presets: plymouth, plymouth_similar_cities, sampling_areas, greater_manchester_las. "
            "Defaults to all of GB if not set."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for this run only.",
    ),
) -> None:
    """Filter OS UPRNs to domestic-only and save result as parquet."""
    settings = Settings(output_dir=output_dir) if output_dir else Settings()

    las = local_authorities.lower() if local_authorities else None

    # Load all UPRN coordinates
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    grid_squares = None
    if las is None:
        print("Creating residential UPRN dataset for all of GB...")
    else:
        print(f"Creating residential UPRN dataset for: {las}")
        grid_squares = config["constant"][las]["grid_squares"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=config["constant"][las]["la_names"]
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    poi_gdf = load_tree_input.load_gdf_poi()
    poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=poi.load_set_non_domestic_poi_categories(),
    )

    layers = {
        f"{layer}_gdf": load_tree_input.load_gdf_os_openmap_local_layer(layer=layer, grid_squares=grid_squares)
        for layer in ["important_building", "railway_station", "building"]
    }

    non_residential_buildings_gdf = non_residential_entities.generate_gdf_non_residential_buildings(
        **layers, poi_gdf=poi_gdf, uprns_gdf=uprns_gdf
    )

    residential_uprns_gdf = uprns.filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=layers["building_gdf"],
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )

    output_cols = ["UPRN", "X_COORDINATE", "Y_COORDINATE", "LATITUDE", "LONGITUDE"]
    if "LAD23CD" in residential_uprns_gdf.columns:
        output_cols += ["LAD23CD", "LAD23NM"]

    df = pl.from_pandas(residential_uprns_gdf[output_cols])

    output_path = settings.resolve_output_path("domestic_uprns.parquet")
    print(f"Saving {len(df):,} domestic UPRNs to: {output_path}")

    # Ensure local output directory exists
    if not output_path.startswith("s3://"):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.write_parquet(output_path)
    else:
        save_utils.save_to_s3(df, output_path)

    print("Done.")


if __name__ == "__main__":
    app()
