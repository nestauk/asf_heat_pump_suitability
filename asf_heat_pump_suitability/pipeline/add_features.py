"""CLI: Add features to domestic UPRNs required for the heat pump suitability decision tree.

Takes the domestic UPRNs parquet produced by ``uprns.py`` and adds:
- ``property_type_flat``: boolean flag indicating whether the UPRN is a flat/apartment
- ``in_block_of_flats``: boolean flag indicating whether the UPRN is in a block of flats
- ``max_contiguous_outdoor_space_area_m2``: estimated max contiguous outdoor space (m2)
- ``total_outdoor_space_area_m2``: estimated total outdoor space (m2)

Reads ``{output_dir}/{lad-slug}/domestic_uprns.parquet`` and writes
``{output_dir}/{lad-slug}/uprns_with_features.parquet`` for each LAD.

Example usage:

    # Single LAD
    uv run python asf_heat_pump_suitability/pipeline/add_features.py --lad manchester

    # Explicit uprns path (overrides default per-LAD path)
    uv run python asf_heat_pump_suitability/pipeline/add_features.py \\
        --lad manchester \\
        --uprns ./outputs/manchester/domestic_uprns.parquet

    # Production run (writes to S3)
    LOCAL_DEV=false uv run python asf_heat_pump_suitability/pipeline/add_features.py --lad manchester
"""

import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
import polars as pl
import typer

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.config.settings import Settings
from asf_heat_pump_suitability.getters import base_getters, get_datasets, load_boundaries, load_tree_input
from asf_heat_pump_suitability.getters.geography_lookup import get_geography_lookup, resolve_lads
from asf_heat_pump_suitability.pipeline.impute import property_type
from asf_heat_pump_suitability.pipeline.model.block_of_flats import feature_engineering, train_model
from asf_heat_pump_suitability.pipeline.transform import outdoor_space, uprns
from asf_heat_pump_suitability.utils import save_utils

logger = logging.getLogger(__name__)

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
    uprns_path: Optional[str] = typer.Option(
        None,
        "--uprns",
        help=(
            "Path to domestic UPRNs parquet (output of uprns.py). "
            "Defaults to {output_dir}/{lad-slug}/domestic_uprns.parquet."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for this run only.",
    ),
) -> None:
    """Add features to domestic UPRNs and save per-LAD results as parquet."""
    settings = Settings(output_dir=output_dir) if output_dir else Settings()

    # Resolve LAD codes
    lookup = get_geography_lookup()
    lad_codes = resolve_lads(lads=lad, utlas=None, combineds=None, lookup=lookup)
    print(f"Processing {len(lad_codes)} LAD(s).")

    # Load model and reference data shared across all LADs
    labelled_df = base_getters.load_df(config["inputs"]["reference"]["manually_labelled_block_of_flats"])
    clf = base_getters.load_pickle(config["outputs"]["models"]["block_of_flats"])

    all_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries()

    # Process each LAD
    for lad_code in lad_codes:
        row = lookup[lookup["lad_code"] == lad_code].iloc[0]
        lad_name = row["lad_name"]
        lad_slug = row["lad_slug"]
        print(f"\n--- Processing LAD: {lad_name} ({lad_code}) ---")

        lad_boundary = all_boundaries_gdf[all_boundaries_gdf["LAD23CD"] == lad_code][["LAD23CD", "LAD23NM", "geometry"]]

        # Resolve uprns path for this LAD
        resolved_uprns_path = (
            uprns_path if uprns_path else settings.resolve_output_path(lad_slug, "domestic_uprns.parquet")
        )
        print(f"Loading domestic UPRNs from: {resolved_uprns_path}")
        uprns_df = base_getters.load_df(resolved_uprns_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])
        uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

        # --- IMPUTE PROPERTY TYPE FLAT ---
        flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
        features_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat"))

        # --- PREDICT BLOCK OF FLATS CLASSIFICATION ---
        building_footprints_gdf = load_tree_input.load_openmap_local_layer(layer="building", lad_boundary=lad_boundary)

        uprn_building_id_dict = uprns.map_dict_uprns_to_building_id(
            uprns_gdf=uprns_gdf, buildings_gdf=building_footprints_gdf, id_col="ID"
        )

        uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)
        building_features_df = feature_engineering.generate_df_features(
            buildings_gdf=building_footprints_gdf,
            uprns_gdf=uprns_gdf,
            id_col="ID",
        )

        features_df = train_model.extend_df_in_block_of_flats_label(
            uprns_df=features_df,
            mapping=uprn_building_id_dict,
            predictions_df=train_model.predict_class_block_of_flats(
                model=clf,
                features_df=building_features_df,
                labelled_df=labelled_df,
                id_col="ID",
            ),
            id_col="ID",
        )

        # --- ESTIMATE OUTDOOR SPACE ---
        print("Loading land registry data...")

        if lad_slug == "plymouth":
            land_parcels_gdf = gpd.read_file(config["inputs"]["geodata"]["plymouth_land_registry"])
        else:
            inspire_files = get_datasets.load_gdf_inspire_land_parcels(
                path=config["inputs"]["inspire"]["file_bounds_ew"]
            )
            inspire_file_names = inspire_files[inspire_files["LAD23NM"] == lad_name]["inspire_file_name"].unique()

            land_parcels_gdf = pd.concat(
                [get_datasets.load_gdf_inspire_land_parcels(path=f"s3://{file}") for file in inspire_file_names],
                ignore_index=False,
            )

        intersection_gdf = outdoor_space.generate_gdf_building_intersections(
            land_parcels_gdf=land_parcels_gdf,
            building_footprints_gdf=building_footprints_gdf,
        )

        outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
            building_intersections_gdf=intersection_gdf, land_parcels_gdf=land_parcels_gdf
        )
        uprns_space_df = outdoor_space.sjoin_df_uprn_to_outdoor_space(
            uprns_gdf=uprns_gdf, outdoor_space_gdf=outdoor_space_gdf
        )
        uprns_space_df = outdoor_space.deduplicate_df_outdoor_space(uprns_space_df)

        features_df = features_df.join(
            uprns_space_df.select(
                [
                    "UPRN",
                    "NATIONALCADASTRALREFERENCE",
                    "max_contiguous_outdoor_space_area_m2",
                    "total_outdoor_space_area_m2",
                ]
            ),
            how="left",
            on="UPRN",
        )

        # --- SAVE OUTPUTS ---
        print(f"Saving {len(features_df):,} UPRNs with features for {lad_name}...")
        out_path = settings.resolve_output_path(lad_slug, "uprns_with_features.parquet")
        save_utils.save_df(features_df, out_path)

    print("\nDone.")


if __name__ == "__main__":
    app()
