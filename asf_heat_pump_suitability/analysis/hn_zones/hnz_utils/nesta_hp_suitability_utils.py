"""
Nesta Heat Pump Suitability Data Utilities

This module provides a utility function to:
- Load and process Nesta’s heat pump suitability scores.
- Filter data by a specified local authority.
- Standardise local authority names to ensure consistency.

**Functions:**
- `filter_la_nesta_hp_scores`: Loads Nesta heat pump suitability data, filters it by local authority,
  and returns a DataFrame of LSOA codes with heat network scores.

This module ensures efficient filtering and processing of suitability data for analysis and comparison.
"""

from typing import Tuple, List
import polars as pl
from config.hnz_config import LA_CORRECTIONS


def filter_la_nesta_hp_scores(
    nesta_hp_suitability_scores: str, local_authority: str
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Load and process Nesta heat pump suitability data for a specific local authority.

    Args:
        nesta_hp_suitability_scores (str): Path to the Nesta heat pump suitability Parquet file with heat network scores.
        local_authority (str): Local authority name to filter the data.

    Returns:
        Tuple[pl.DataFrame, List[str]]:
            - DataFrame containing filtered LSOA codes and 'HN_N_avg_score_weighted' column.
            - List of unique LSOA codes within the local authority.
    """
    # Load the data from the Parquet file
    hp_scores_df = pl.read_parquet(nesta_hp_suitability_scores)

    # Rename 'lsoa' to 'LSOA21CD' if needed to match the GeoDataFrame
    if "lsoa" in hp_scores_df.columns:
        hp_scores_df = hp_scores_df.rename({"lsoa": "LSOA21CD"})

    # Filter by local authority and select relevant columns
    transformed_local_authority = LA_CORRECTIONS.get(local_authority, local_authority)
    la_filtered_scores = hp_scores_df.filter(
        pl.col("lsoa_name").str.starts_with(transformed_local_authority)
    ).select(["LSOA21CD", "HN_N_avg_score_weighted"])

    # Extract unique LSOA codes for LA
    la_unique_lsoas = la_filtered_scores["LSOA21CD"].unique().to_list()

    return la_filtered_scores, la_unique_lsoas
