import pathlib
import s3fs
import pandas as pd
import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config

BLOCK_OF_FLATS_ARCHETYPES = [
    "TB",
    "BF",
    "BP",
    "TN",
    "TC",
    "SF",
    "TT",
    "AS",
    "IW",
    "NH",
]

NOT_BLOCKS_ARCHETYPES = [
    "TS",
    "TE",
    "TF",
    "TY",
    "BB",
    "WB",
    "FC",
    "RA",
    "MA",
    "MT",
    "OF",
    "CO",
]

EXCLUDED_ARCHETYPES = ["DE", "UL"]


def load_gdf_unprocessed_labelled() -> gpd.GeoDataFrame:
    """
    Load unprocessed manually labelled data for block of flats classifier model and concatenate into a single GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: unprocessed manually labelled data
    """
    fs = s3fs.S3FileSystem()
    dir_path = pathlib.Path(config["output"]["model"]["training_data"]["labelled_dir"])
    paths = [f for f in fs.glob(f"{dir_path}/*.kml")]
    [print(f"Loading unprocessed labelled data from: {path}") for path in paths]
    return pd.concat([gpd.read_file(path) for path in paths])


def extract_df_labelled_data(
    gdf: gpd.GeoDataFrame, id_str: str = "building_id"
) -> pl.DataFrame:
    """
    Extract label, confidence, URL, and labeller from manually labelled sample data from .kml format.

    Args:
        gdf (gpd.GeoDataFrame): manually labelled sample data
        id_str (str): name of building ID substring to search for in 'Description' column of gdf

    Returns:
        pl.DataFrame: extracted information for manually labelled sample data
    """
    gdf[id_str] = gdf.description.str.extract(rf"{id_str}: (.+) -")
    gdf["label"] = gdf.Name.str[:2].str.upper()
    gdf["confidence"] = gdf["Name"].str[-1:]
    gdf["url"] = gdf.description.str.extract(r"Location: (.+) -")

    cols = [id_str, "label", "confidence", "url"]
    df = pl.from_pandas(gdf[cols])

    assert df.null_count().sum_horizontal()[
        0
    ], "There are unexpected null values in the processed labelled sample. Please check processing has worked."

    all_labels = BLOCK_OF_FLATS_ARCHETYPES + NOT_BLOCKS_ARCHETYPES + EXCLUDED_ARCHETYPES
    unexpected = set(df["label"]).difference(set(all_labels))
    assert (
        not unexpected
    ), f"There are unexpected archetype labels in the labelled sample: {unexpected}."

    return df.with_columns(
        pl.when(pl.col("label") == "UL")
        .then(pl.lit(None))
        .otherwise(pl.col("confidence"))
        .alias("confidence")
    )


def compare_df_labellers(
    labelled_df: pl.DataFrame, unlabelled_df: pl.DataFrame, id_str: str
) -> pl.DataFrame:
    labelled_df = labelled_df.join(
        unlabelled_df.select([id_str, "split", "labeller", "secondary_labeller"]),
        how="left",
        on=id_str,
    ).with_columns(
        pl.when(pl.col("label").is_in(BLOCK_OF_FLATS_ARCHETYPES))
        .then(True)
        .when(pl.col("label").is_in(NOT_BLOCKS_ARCHETYPES))
        .then(False)
        .otherwise(None)
        .alias("block_of_flats"),
    )

    # Filter to duplicated building IDs (meaning the building has been labelled by two different people)
    labelled_df = (
        labelled_df.with_columns(pl.col(id_str).is_duplicated().alias("duplicate"))
        .filter(pl.col("duplicate"))
        .group_by(id_str, maintain_order=True)
        .agg(
            # Aggregate the multiple labels and other information into a list per building
            pl.col("label"),
            pl.col("block_of_flats"),
            pl.col("confidence"),
            pl.col("labeller"),
            pl.col("url").first(),
        )
        .with_columns(
            # Convert lists to structs and then unnest to create one row per building with information from both labellers
            pl.col("label").list.to_struct(
                fields=["sublabel_labeller1", "sublabel_labeller2"]
            ),
            pl.col("block_of_flats").list.to_struct(
                fields=["label_labeller1", "label_labeller2"]
            ),
            pl.col("confidence").list.to_struct(
                fields=["confidence_labeller1", "confidence_labeller1"]
            ),
            pl.col("labeller").list.to_struct(fields=["labeller1", "labeller2"]),
        )
        .unnest(columns=["label", "block_of_flats", "confidence", "labeller"])
        .with_columns(
            (pl.col("label_labeller1") == pl.col("label_labeller2")).alias(
                "agree_label"
            ),
            (pl.col("sublabel_labeller1") == pl.col("sublabel_labeller2")).alias(
                "agree_sublabel"
            ),
        )
    )

    return labelled_df


def transform_df_labelled_data(
    labelled_df: pl.DataFrame, unlabelled_df: pl.DataFrame, id_str: str
) -> pl.DataFrame:
    """
    Prepare labelled data for modelling inputs.
    """
    labelled_df = labelled_df.join(
        unlabelled_df.select([id_str, "split", "labeller", "secondary_labeller"]),
        how="left",
        on=id_str,
    ).with_columns(
        pl.when(pl.col("label").is_in(BLOCK_OF_FLATS_ARCHETYPES))
        .then(True)
        .when(pl.col("label").is_in(NOT_BLOCKS_ARCHETYPES))
        .then(False)
        .otherwise(None)
        .alias("block_of_flats"),
    )
    return labelled_df
