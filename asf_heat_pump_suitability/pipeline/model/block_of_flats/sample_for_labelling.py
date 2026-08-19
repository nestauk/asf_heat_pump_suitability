"""
Create a sample of buildings containing flats for manual labelling to use in model training.
"""

if "name" == "__main__":
    import polars as pl
    from asf_heat_pump_suitability.getters import load_data
    from asf_heat_pump_suitability.pipeline.impute import property_type

    # Load our domestic UPRNs from processing
    domestic_uprns = set(pl.read_parquet("path", columns="UPRN")["UPRN"])

    # Load the lookup with all the additional data
    uprns_df = load_data.load_df_uprn_lookup().filter(
        pl.col("UPRN").is_in(domestic_uprns)
    )
    flat_uprns = property_type.impute_set_flat_properties(
        uprns_df, x_col="GRIDGB1E", y_col="GRIDGB1N"
    )
    uprns_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("is_flat"))

    flats_df = uprns_df.filter(pl.col("is_flat"))


# STEP-BY-STEP APPROACH
# 1. [DONE] Label all UPRNs as flats or not.
# 2. [DONE] Identify all buildings containing flats.
# 3. [DONE] Label buildings with rural/urban indicator
# 4. [DONE] Label buildings with IMD decile
# 5. Label buildings with country
# 7. Get count of flats per building
# 8. Label buildings with grouped-count
# 9. Sample based on rural/urban indicator; IMD decile; country; and grouped count
# 10. Enrich sample with additional data:
# - URL
# - Count of UPRNs per building
# 11. Convert to kml file
