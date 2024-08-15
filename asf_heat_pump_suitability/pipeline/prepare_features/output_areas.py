import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import get_datasets


def transform_df_ons_pd(
    pcd_col: str = "pcd",
    ruc_col: str = "ru11ind",
    lad_col: str = "oslaua",
    use_cols: list = [
        "pcd",
        "lsoa11",
        "msoa11",
        "lsoa21",
        "msoa21",
        "ru11ind",
        "oslaua",
    ],
) -> pl.DataFrame:
    """
    Load and transform ONS postcode directory dataset: standardise postcode; clean output area columns; add new
    `country_code` column; map rural-urban indicators.

    Args
        pcd_col (str): name of column containing postcodes. Default "pcd".
        ruc_col (str): name of column containing rural-urban classification codes. Default "ru11ind".
        lad_col (str): name of column containing Local Authority District (LAD) codes. Default "oslaua".
        use_cols (list): columns to import. Default ["pcd", "lsoa11", "msoa11", "lsoa21", "msoa21", "ru11ind", "oslaua"].

    Returns
        pl.DataFrame: processed ONS postcode directory dataset
    """
    df = get_datasets.get_df_ons_pd(columns=use_cols).rename(
        mapping={lad_col: "lad_code"}
    )
    df = standardise_col_postcode(df, pcd_col=pcd_col)
    df = _clean_col_output_area(df, area_type="lsoa")
    df = _clean_col_output_area(df, area_type="msoa")
    df = _create_col_country_code(df)
    df = map_cols_ruc(df, ruc_col=ruc_col)

    return df


def standardise_col_postcode(df: pl.DataFrame, pcd_col: str) -> pl.DataFrame:
    """
    Standardise postcode column of a dataset: uppercase all letters and remove spaces.

    Args
        df (pl.DataFrame): dataset
        pcd_col (str): name of column containing postcodes

    Returns
        pl.DataFrame: dataset with standardised postcode column
    """

    df = df.with_columns(
        pl.col(pcd_col).str.to_uppercase().str.replace(r"\s+", "").alias("POSTCODE")
    )

    return df


def map_cols_ruc(df: pl.DataFrame, ruc_col: str) -> pl.DataFrame:
    """
    Add new columns to dataframe with rural-urban classification text descriptions mapped from rural-urban
    classification code. Adds: two-fold classification; 10-fold classification for England and Wales only; 8-fold
    classification for Scotland only.

    Args
        df (pl.DataFrame): dataset
        pcd_col (str): name of column containing rural-urban classification codes

    Returns
        pl.DataFrame: dataset with additional rural-urban classification description columns
    """
    df = df.with_columns(
        [
            pl.col(ruc_col)
            .replace(config["mapping"]["ruc_two_fold"], default=None)
            .alias("ruc_two_fold"),
            pl.col(ruc_col)
            .replace(config["mapping"]["ruc_EW_ten_fold"], default=None)
            .alias("ruc_EW_ten_fold"),
            pl.col(ruc_col)
            .replace(config["mapping"]["ruc_S_eight_fold"], default=None)
            .alias("ruc_S_eight_fold"),
        ]
    )
    return df


def _clean_col_output_area(df: pl.DataFrame, area_type: str) -> pl.DataFrame:
    """
    Create new clean output area column in dataset using existing output area columns from 2011 and 2021 census.

    Args
        df (pl.DataFrame): dataset with area columns
        area_type (str): type of output area, e.g. `oa`, `lsoa`, `msoa`

    Returns
        pl.DataFrame: dataset with new output area column
    """
    df = df.with_columns(pl.col([f"{area_type}11", f"{area_type}21"]).replace("", None))
    df = df.with_columns(
        pl.col(f"{area_type}21").fill_null(pl.col(f"{area_type}11")).alias(area_type)
    )

    return df


def _create_col_country_code(df: pl.DataFrame, code_col: str = "lsoa") -> pl.DataFrame:
    """
    Create new `country_code` column derived from area code.

    Args
        df (pl.DataFrame): dataset with area code column
        code_col (str): name of column containing area codes

    Returns
        df (pl.DataFrame): dataset with new `country_code` column
    """
    return df.with_columns(pl.col(code_col).str.slice(0, 1).alias("country_code"))
