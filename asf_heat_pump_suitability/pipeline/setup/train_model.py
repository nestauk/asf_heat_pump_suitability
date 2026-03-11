"""CLI: Train the block-of-flats classification model and save to S3.

This is a setup companion script — run it once to train the Random Forest
classifier that ``add_features.py`` uses to predict whether a building is
a block of flats or not.

Requires domestic UPRNs parquet and manually-labelled building data.

Example usage:

    uv run python asf_heat_pump_suitability/pipeline/setup/train_model.py \\
        --uprns s3://asf-local-heat-planning-tool/outputs/sampling_areas_residential_uprns.parquet \\
        --labelled-data s3://asf-local-heat-planning-tool/inputs/reference/manually_labelled_block_of_flats.parquet \\
        --save
"""

import typer

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, load_tree_input
from asf_heat_pump_suitability.pipeline.impute import property_type
from asf_heat_pump_suitability.pipeline.model.block_of_flats import feature_engineering, train_model
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.utils import save_utils

app = typer.Typer(help=__doc__)


@app.command()
def main(
    uprns_path: str = typer.Option(
        ...,
        "--uprns",
        help="Path to domestic UPRN dataset with X and Y coordinates in parquet.",
    ),
    labelled_data: str = typer.Option(
        ...,
        "--labelled-data",
        help=(
            "Path to labelled data to train binary classification model on in parquet format. "
            "Requires a boolean 'block_of_flats' column and a building ID column with one row per building. "
            "Note: the UPRNs file must contain all domestic UPRNs within the area(s) that labelled_data samples from."
        ),
    ),
    save: bool = typer.Option(False, "--save", help="Save trained model to S3."),
) -> None:
    """Train the block-of-flats Random Forest classifier and optionally save it to S3."""
    print(f"Loading domestic UPRNs from: {uprns_path}")
    uprns_df = base_getters.load_df(uprns_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])

    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building",
        grid_squares=config["constant"]["sampling_areas"]["grid_squares"],
    )

    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)

    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    print(f"Loading labelled data from: {labelled_data}")
    labelled_df = base_getters.load_df(labelled_data)
    model_df = labelled_df.join(building_features_df, how="left", on="ID")

    model = train_model.train_eval_rfc_block_of_flats_classifier(
        df=model_df,
        id_col="ID",
        features=train_model.FEATURES,
        target="block_of_flats",
        param_search="default",
    )

    if save:
        save_utils.save_model_to_pkl(model, config["outputs"]["models"]["block_of_flats"])
    else:
        print("Model trained. Use --save to persist to S3.")


if __name__ == "__main__":
    app()
