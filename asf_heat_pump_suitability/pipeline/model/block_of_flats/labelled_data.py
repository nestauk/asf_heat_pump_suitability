"""
Functions to load and process manually labelled data and prepare it for use in binary classification model training.
"""

import pathlib
import s3fs
import numpy as np
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


def load_gdf_unprocessed_labelled_data() -> gpd.GeoDataFrame:
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
        id_str (str): name of building ID substring to search for in 'Description' column of gdf.  Default 'building_id'.

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
        .alias("confidence"),
        pl.when(pl.col("label").is_in(BLOCK_OF_FLATS_ARCHETYPES))
        .then(True)
        .when(pl.col("label").is_in(NOT_BLOCKS_ARCHETYPES))
        .then(False)
        .otherwise(None)
        .alias("block_of_flats"),
    )


def compare_tuple_labellers(
    labelled_df: pl.DataFrame, unlabelled_df: pl.DataFrame, id_str: str = "building_id"
) -> tuple:
    """
    Compare labels of buildings which have received two labels to assess agreement levels between labellers.

    Args:
        labelled_df (pl.DataFrame): dataframe containing all labelled samples from all labellers
        unlabelled_df (pl.DataFrame): dataframe containing sample data with `labeller` and `secondary_labeller` columns
        id_str (str): name of column containing building ID in `labelled_df` and `unlabelled_df`. Default 'building_id'.

    Returns:
        tuple: (building IDs where labellers agree on the binary class, building IDs where labellers disagree)
    """
    double_labelled_df = (
        # Join labellers onto labelled buildings
        labelled_df.join(
            unlabelled_df.select([id_str, "labeller", "secondary_labeller"]),
            how="left",
            on=id_str,
            # Filter to duplicated building IDs (meaning the building has been labelled by two different people)
        )
        .with_columns(pl.col(id_str).is_duplicated().alias("duplicate"))
        .filter(pl.col("duplicate"))
        .group_by(id_str, maintain_order=True)
        .agg(
            # Aggregate the multiple labels and other information into a list per building
            pl.col("block_of_flats"),
            pl.col("confidence"),
            pl.col("labeller").first(),
            pl.col("secondary_labeller").first(),
        )
        .with_columns(
            # Convert lists to structs and then unnest to create one row per building with information from both labellers
            pl.col("block_of_flats").list.to_struct(
                fields=["label_labeller1", "label_labeller2"]
            ),
            pl.col("confidence").list.to_struct(
                fields=["confidence_labeller1", "confidence_labeller1"]
            ),
        )
        .unnest(columns=["block_of_flats", "confidence"])
        .with_columns(
            (pl.col("label_labeller1") == pl.col("label_labeller2")).alias(
                "agree_label"
            ),
        )
    )

    print_labeller_agreement_matrix(df=double_labelled_df)

    # Building IDs where the labels (of the aggregated four categories) match between labellers
    agree_buildings_ids = (
        double_labelled_df.filter(pl.col("agree_label"))[id_str].unique().to_list()
    )

    # Building IDs where the labels (of the aggregated four categories) do not match between labellers
    disagree_building_ids = (
        double_labelled_df.filter(~pl.col("agree_label"))[id_str].unique().to_list()
    )

    return agree_buildings_ids, disagree_building_ids


def print_labeller_agreement_matrix(df: pl.DataFrame) -> None:
    """
    Print the agreement matrix for each labeller pair.

    Args:
        df (pl.DataFrame): samples with two labels from different labellers. Must have `labeller1`; `labeller2`; and `agree_label` columns.

    Returns:
        None
    """
    labellers = df["labeller1"].unique().to_list()

    agreement = []

    for l1 in labellers:
        labeller1_df = df.filter(pl.col("labeller1") == l1)
        labellers_2 = labeller1_df["labeller2"].unique().to_list()
        for l2 in labellers_2:
            print(
                f"\n\nPrimary labeller: {l1}; secondary labeller: {l2};\nAgreement matrix:\n"
            )
            print(
                labeller1_df.filter(pl.col("labeller2") == l2)[
                    "agree_label"
                ].value_counts()
            )
            agreement_matrix = labeller1_df.filter(pl.col("labeller2") == l2)[
                "agree_label"
            ].value_counts(normalize=True)
            agreement.append(
                agreement_matrix.filter(pl.col("agree_label"))["proportion"][0]
            )

    print(
        f"\n\nAverage agreement between labellers: {round(np.mean(agreement) * 100, 2)}%"
    )


def transform_df_labelled_data(
    labelled_df: pl.DataFrame, unlabelled_df: pl.DataFrame, id_str: str = "building_id"
) -> pl.DataFrame:
    """
    Prepare labelled data for training model: assign train / test label; remove buildings where labellers disagree on
    binary class; set confidence to 1 where labellers agree on binary class; remove buildings with labels intended for
    exclusion.

    Args:
        labelled_df (pl.DataFrame): dataframe containing all labelled samples from all labellers
        unlabelled_df (pl.DataFrame): dataframe containing sample data with `labeller` and `secondary_labeller` columns
        id_str (str): name of column containing building ID in `labelled_df` and `unlabelled_df`. Default 'building_id'.

    Returns:
        pl.DataFrame: labelled data ready for training binary classifier
    """
    agree_ids, disagree_ids = compare_tuple_labellers(
        labelled_df=labelled_df, unlabelled_df=unlabelled_df, id_str=id_str
    )

    return (
        # Add train / test split label
        labelled_df.join(unlabelled_df.select([id_str, "split"]))
        .filter(
            # Remove buildings where labellers disagree
            ~pl.col(id_str).is_in(disagree_ids),
        )
        .with_columns(
            # For buildings where labellers agree, set the confidence to 1, otherwise retain original confidence
            pl.when(pl.col(id_str).is_in(agree_ids))
            .then(pl.lit("1"))
            .otherwise(pl.col("confidence"))
            .alias("confidence")
        )
        .with_columns(
            pl.col("confidence")
            .cast(pl.Int8)
            .alias("confidence")
            # Drop duplicate building IDs (i.e. buildings with two labellers)
            # Note that this affects the distribution of subclasses within groups which may be different between labellers
        )
        .unique(subset=[id_str], keep="any")
        .filter(
            # Remove buildings which are excluded from training data
            ~pl.col("label").is_in(EXCLUDED_ARCHETYPES)
        )
    )
