import geopandas as gpd
import polars as pl

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


def extract_df_labelled_data(gdf: gpd.GeoDataFrame, id_str: str) -> pl.DataFrame:
    """
    Extract label, confidence, URL, and labeller from manually labelled sample data from .kml format.

    Args:
        gdf (gpd.GeoDataFrame): manually labelled sample data
        id_str (str): name of building ID substring to search for in 'Description' column of gdf

    Returns:
        pl.DataFrame: extracted information for manually labelled sample data
    """
    gdf[id_str] = gdf.description.str.extract(r"building_id: (.+) -")
    gdf["label"] = gdf.Name.str[:2].str.upper()
    gdf["confidence"] = gdf["Name"].str[-1:]
    gdf["url"] = gdf.description.str.extract(r"Location: (.+) -")

    cols = [id_str, "label", "confidence", "url"]
    df = pl.from_pandas(gdf[cols])

    assert df.null_count().sum_horizontal()[
        0
    ], "There are unexpected null values in the processed labelled sample. Please check processing has worked."

    all_labels = BLOCK_OF_FLATS_ARCHETYPES + NOT_BLOCKS_ARCHETYPES + EXCLUDED_ARCHETYPES
    assert not set(df["label"]).difference(
        set(all_labels)
    ), "There are unexpected archetype labels in the labelled sample."

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
