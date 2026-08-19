"""
Create a sample of buildings containing flats for manual labelling to use in model training.
"""

if "name" == "__main__":
    import polars as pl
    from asf_heat_pump_suitability.getters import load_data, load_geodata
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns

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

    # Load building footprints and map to UPRNs
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(layer="building")
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df, usecols=["UPRN", "is_flat"])
    uprn_building_mapping = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf, id_col="ID"
    )
    uprns_df = uprns_df.with_columns(
        # Map building IDs to the UPRNs they contain
        pl.col("UPRN")
        .replace_strict(uprn_building_mapping, default=None)
        .alias("building_id")
    )

    flats_df = uprns_df.filter(pl.col("is_flat"))


# STEP-BY-STEP APPROACH
# 1. [DONE] Label all UPRNs as flats or not.
# 2. [DONE] Identify all buildings containing flats.
# 3. [DONE] Label buildings with rural/urban indicator
# 4. [DONE] Label buildings with IMD decile
# 5. [DONE] Label buildings with country
# 7. Get count of flats per building
# 8. Label buildings with grouped-count
# 9. Sample based on rural/urban indicator; IMD decile; country; and grouped count
# 10. Enrich sample with additional data:
# - URL
# - Count of UPRNs per building
# 11. Convert to kml file
