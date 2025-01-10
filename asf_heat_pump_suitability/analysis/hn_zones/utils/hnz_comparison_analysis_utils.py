"""
Heat Network Zone (HNZ) Analysis Utilities

This module provides utility functions for analyzing and comparing
DESNZ heat network zones with Nesta's heat network suitability data.

**Key Features:**
- Adds a `DESNZ_pilot_fraction` column to estimate the fraction of an LSOA covered by heat network zones.
- Computes average suitability scores for LSOAs within and outside heat network zones.
- Calculates Mean Absolute Error (MAE) between DESNZ pilot scores and Nesta suitability scores.

**Functions:**
- `add_DESNZ_pilot_fraction`: Merges heat network zone data with suitability scores and computes the fraction covered.
- `calculate_average_scores_for_thresholds`: Computes average suitability scores for different thresholds of coverage.
- `calculate_mae_for_pilot_score`: Calculates MAE for LSOAs inside and outside pilot heat network zones.
- `calculate_mae_for_all`: Computes overall MAE between DESNZ and Nesta scores.

This module supports comparative spatial analysis of heat network zones and heat network suitability data.
"""

from typing import Tuple, List, Optional
import geopandas as gpd
import polars as pl
import logging


def add_DESNZ_pilot_fraction(
    la_hp_suitability_scores: pl.DataFrame,
    joined_gdf: gpd.GeoDataFrame,
    optional_threshold: Optional[float] = 0,
) -> Tuple[pl.DataFrame, float, float]:
    """
    Add a 'DESNZ_pilot_fraction' column to the Nesta heat pump suitability data using the fraction of area covered by heat network zones and compute the average Nesta heat network score per LA.

    Args:
        la_hp_suitability_scores (pl.DataFrame): Data containing 'LSOA21CD' and 'HN_N_avg_score_weighted' columns for a single LA.
        joined_gdf (gpd.GeoDataFrame): GeoDataFrame containing 'LSOA21CD' and 'fraction_covered' columns.
        optional_threshold (Optional[float]): Optional threshold for fraction of heat network zone area contained within LA. Defaults to 0.

    Returns:
        Tuple[pl.DataFrame, float, float]:
            - Updated DataFrame with 'DESNZ_pilot_fraction' column added (fraction of local authority area covered by heat network zones).
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_fraction' is non-zero.
            - Average 'HN_N_avg_score_weighted' for rows where 'DESNZ_pilot_fraction' is zero.
    """
    # Convert joined_gdf to a Polars DataFrame
    joined_df = pl.DataFrame(
        {
            "LSOA21CD": joined_gdf["LSOA21CD"],
            "fraction_covered": joined_gdf["fraction_covered"],
        }
    )

    # Merge the fraction_covered into la_hp_suitability_scores
    la_hp_suitability_scores = la_hp_suitability_scores.join(
        joined_df, on="LSOA21CD", how="left"
    )

    # Fill NaN values in fraction_covered with 0
    la_hp_suitability_scores = la_hp_suitability_scores.with_columns(
        pl.col("fraction_covered").fill_null(0).alias("DESNZ_pilot_fraction")
    ).drop("fraction_covered")

    # Calculating the average Nesta heat network score for non-zero and zero DESNZ pilot scores
    avg_hn_score_pilot_nonzero = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores,
        hn_zones=True,
        optional_threshold=optional_threshold,
    )
    avg_hn_score_pilot_zero = _calculate_hn_pilot_average_score(
        la_hp_suitability_scores, hn_zones=False
    )
    return la_hp_suitability_scores, avg_hn_score_pilot_nonzero, avg_hn_score_pilot_zero


def calculate_average_scores_for_thresholds(
    la_hp_suitability_scores: pl.DataFrame, thresholds: List[float]
) -> pl.DataFrame:
    """
    Calculate the average 'HN_N_avg_score_weighted' for various thresholds of 'DESNZ_pilot_fraction'.

    Args:
        la_hp_suitability_scores (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'HN_N_avg_score_weighted' columns.
        thresholds (List[float]): List of thresholds to evaluate.

    Returns:
        pl.DataFrame: DataFrame with columns 'DESNZ_pilot_fraction_threshold' and 'HN_N_avg_score_weighted'.
    """
    results = []
    for threshold in thresholds:
        avg_score = _calculate_hn_pilot_average_score(
            la_hp_suitability_scores, hn_zones=True, optional_threshold=threshold
        )
        results.append(
            {
                "DESNZ_pilot_fraction_threshold": threshold,
                "HN_N_avg_score_weighted": avg_score,
            }
        )
    results_df = pl.DataFrame(results)
    return results_df


def _calculate_hn_pilot_average_score(
    la_hp_suitability_scores: pl.DataFrame,
    hn_zones: bool,
    optional_threshold: Optional[float] = 0,
) -> float:
    """
    Calculate the average Nesta Heat Network score for LSOAs in (`DESNZ_pilot_fraction > optional_threshold`) or not in (`DESNZ_pilot_fraction == 0`) DESNZ heat network pilot areas.

    Args:
        la_hp_suitability_scores (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'HN_N_avg_score_weighted' columns for a LA.
        hn_zones (bool): If True, calculate average Nesta heat network score for LSOAs in DESNZ heat network zones. Set to False to calculate the average score for LSOAs not in heat network zones.
        optional_threshold (Optional[float]): The threshold value for 'DESNZ_pilot_fraction'. Defaults to 0. Range: 0-1.

    Returns:
        float: Average 'HN_N_avg_score_weighted' for the filtered rows.
    """
    if hn_zones:
        # Filter for LSOAs in DESNZ heat network zones
        filtered_la_hp_suitability_scores = la_hp_suitability_scores.filter(
            pl.col("DESNZ_pilot_fraction") > optional_threshold
        )
    else:
        # Filter for LSOAs not in DESNZ heat network zones
        filtered_la_hp_suitability_scores = la_hp_suitability_scores.filter(
            pl.col("DESNZ_pilot_fraction") == 0
        )

    # Calculate the average score
    avg_score = filtered_la_hp_suitability_scores["HN_N_avg_score_weighted"].mean()
    return avg_score


def calculate_mae_for_pilot_score(
    hp_suitability_scores_with_desnz: pl.DataFrame,
    hn_zones: bool,
    optional_threshold: Optional[float] = 0,
) -> float:
    """
    Calculate the Mean Absolute Error (MAE) for entries in or not in DESNZ heat network zones.

    Args:
        hp_suitability_scores_with_desnz (pl.DataFrame): DataFrame containing 'DESNZ_pilot_fraction' and 'absolute_error' columns.
        hn_zones (bool): If True, calculate MAE for entries in DESNZ heat network zones. Set to False to calculate MAE for entries not in heat network zones.
        optional_threshold (Optional[float]): The threshold value for 'DESNZ_pilot_fraction'. Defaults to 0. Range: 0-1.

    Returns:
        float: The MAE for the specified condition.
    """
    if hn_zones:
        # Filter for entries in DESNZ heat network zones
        filtered_df = hp_suitability_scores_with_desnz.filter(
            pl.col("DESNZ_pilot_fraction") > optional_threshold
        )
    else:
        # Filter for entries not in DESNZ heat network zones
        filtered_df = hp_suitability_scores_with_desnz.filter(
            pl.col("DESNZ_pilot_fraction") == 0
        )

    # Calculate the MAE
    mae = filtered_df["absolute_error"].mean()
    return mae


def calculate_mae_for_all(
    hp_suitability_scores: pl.DataFrame, desnz_col: str, nesta_hn_score_col: str
) -> Tuple[pl.DataFrame, float]:
    """
    Calculate the Mean Absolute Error (MAE) between the DESNZ pilot score and Nesta's heat network suitability score, add the absolute error column to the DataFrame,
    and log the result.

    Args:
        hp_suitability_scores (pl.DataFrame): DataFrame containing the DESNZ pilot (actual) and Nesta heat network suitability score (predicted) columns.
        desnz_col (str): Name of the column with DESNZ pilot (actual) values.
        nesta_hn_score_col (str): Name of the column with Nesta heat network suitability (predicted) values.

    Returns:
        Tuple[pl.DataFrame, float]:
            - Updated DataFrame with 'absolute_error' column added.
            - The Mean Absolute Error (MAE) between the actual and predicted columns.
    """
    # Calculate absolute error and add it as a new column
    hp_suitability_scores_with_error = hp_suitability_scores.with_columns(
        (pl.col(desnz_col) - pl.col(nesta_hn_score_col)).abs().alias("absolute_error")
    )

    # Calculate the mean of the absolute error
    mae = hp_suitability_scores_with_error["absolute_error"].mean()
    logging.info(
        f"Mean Absolute Error (MAE) for {desnz_col} vs {nesta_hn_score_col}: {mae}"
    )

    return hp_suitability_scores_with_error, mae
