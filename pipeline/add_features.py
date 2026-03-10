"""CLI: Add features to domestic UPRNs required for the heat pump suitability decision tree.

Takes the domestic UPRNs parquet produced by ``uprns.py`` and adds:
- ``property_type_flat``: boolean flag indicating whether the UPRN is a flat/apartment
- ``in_block_of_flats``: boolean flag indicating whether the UPRN is in a block of flats
- ``max_contiguous_outdoor_space_area_m2``: estimated max contiguous outdoor space (m2)
- ``total_outdoor_space_area_m2``: estimated total outdoor space (m2)

Example usage:

    # Default — reads domestic_uprns.parquet from OUTPUT_DIR, writes uprns_with_features.parquet
    uv run python pipeline/add_features.py --local-authorities greater_manchester_las

    # Explicit uprns path
    uv run python pipeline/add_features.py \\
        --uprns ./outputs/domestic_uprns.parquet \\
        --local-authorities greater_manchester_las

    # Production run (writes to S3)
    LOCAL_DEV=false uv run python pipeline/add_features.py
"""

import os
from typing import Optional

import geopandas as gpd
import pandas as pd
import polars as pl
import typer

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.config.settings import Settings
from asf_heat_pump_suitability.getters import base_getters, get_datasets, load_tree_input
from asf_heat_pump_suitability.pipeline.impute import property_type
from asf_heat_pump_suitability.pipeline.model.block_of_flats import feature_engineering, train_model
from asf_heat_pump_suitability.pipeline.transform import outdoor_space, uprns
from asf_heat_pump_suitability.utils import save_utils

app = typer.Typer(help=__doc__)


@app.command()
def main(
    uprns_path: Optional[str] = typer.Option(
        None,
        "--uprns",
        help=("Path to domestic UPRNs parquet (output of uprns.py). Defaults to OUTPUT_DIR/domestic_uprns.parquet."),
    ),
    local_authorities: Optional[str] = typer.Option(
        None,
        "--local-authorities",
        help=(
            "Local authority preset or space-separated LA codes. "
            "Available presets: plymouth, plymouth_similar_cities, sampling_areas, greater_manchester_las."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override OUTPUT_DIR env var for this run only.",
    ),
) -> None:
    """Add features to domestic UPRNs and save result as parquet."""
    settings = Settings(output_dir=output_dir) if output_dir else Settings()

    las = local_authorities.lower() if local_authorities else None

    # Resolve uprns path
    if uprns_path is None:
        uprns_path = settings.resolve_output_path("domestic_uprns.parquet")

    # Load UPRN data
    print(f"Loading domestic UPRNs from: {uprns_path}")
    uprns_df = pl.read_parquet(uprns_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])

    # Get geopoints of UPRNs
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # --- IMPUTE PROPERTY TYPE FLAT ---
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    features_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat"))

    # --- PREDICT BLOCK OF FLATS CLASSIFICATION ---
    grid_squares = config["constant"][las]["grid_squares"] if las else None
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=grid_squares
    )

    uprn_building_id_dict = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=building_footprints_gdf, id_col="ID"
    )

    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    labelled_df = pl.read_parquet(
        "s3://asf-heat-pump-suitability/local_heat_planning/inputs/processed/manually_labelled_block_of_flats.parquet"
    )
    clf = base_getters.load_pickle(config["output"]["save_as"]["model"]["block_of_flats_model"])
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

    if las == "plymouth":
        land_parcels_gdf = gpd.read_file(
            "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Plymouth_Land_Registry_Cadastral_Parcels.gml"
        )
    else:
        inspire_file_names = get_datasets.load_gdf_inspire_land_parcels(
            path="s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/inspire_file_bounds_EW.geojson"
        )
        inspire_file_names = inspire_file_names[
            inspire_file_names["LAD23NM"].isin(config["constant"][las]["la_names"])
        ]["inspire_file_name"].unique()

        land_parcels_gdf = pd.concat(
            [get_datasets.load_gdf_inspire_land_parcels(path=f"s3://{file}") for file in inspire_file_names],
            ignore_index=False,
        )

    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=grid_squares
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
    output_path = settings.resolve_output_path("uprns_with_features.parquet")
    print(f"Saving {len(features_df):,} UPRNs with features to: {output_path}")

    if not output_path.startswith("s3://"):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        features_df.write_parquet(output_path)
    else:
        save_utils.save_to_s3(features_df, output_path)

    print("Done.")


if __name__ == "__main__":
    app()
