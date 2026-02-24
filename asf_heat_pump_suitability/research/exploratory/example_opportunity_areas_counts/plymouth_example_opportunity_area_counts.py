"""
Script to use EPC data to get value counts of TENURE, ATTACHMENT TYPE, and EPC RATING for
3 example opportunity areas in Plymouth.

To run:
python asf_heat_pump_suitability/research/exploratory/example_opportunity_areas_counts/plymouth_example_opportunity_area_counts.py
"""

if __name__ == "__main__":
    import polars as pl
    import geopandas as gpd
    from asf_heat_pump_suitability.getters import load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns

    # Load EPC
    print("Loading deduplicated EPC data...")
    raw_epc_df = pl.read_parquet(
        "s3://asf-daps/lakehouse/2025_Q1/processed/epc/deduplicated/processed_dedupl-0.parquet"
    )

    # Load UPRNs
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    # Load opportunity areas
    print("Loading opportunity areas...")
    areas_gdf = gpd.read_file(
        "s3://asf-heat-pump-suitability/dump/example_opportunity_areas_plymouth.kml"
    )
    areas_gdf = areas_gdf.to_crs(epsg=27700)

    keep_cols = [
        "UPRN",
        "PROPERTY_TYPE",
        "BUILT_FORM",
        "TENURE",
        "CURRENT_ENERGY_RATING",
        "ENERGY_RATING_CAT",
    ]

    print("Processing EPC data...")
    # Filter to valid UPRNs
    epc_df = (
        raw_epc_df.select(keep_cols)
        .with_columns(
            # Remove any invalid UPRNs (i.e. those IDs which are generated in EPC preprocessing generated from concatenating building ref number and address)
            # These are not true UPRNs that can be used in joins across other datasets
            pl.col("UPRN")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64)
            .alias("processed_UPRN")
        )
        .drop_nulls(subset="processed_UPRN")
        .with_columns(
            pl.col("PROPERTY_TYPE").cast(pl.String),
            pl.col("BUILT_FORM").cast(pl.String),
        )
        .with_columns(
            # Reassign enclosed terrace categories and set 'flat' as an attachment type
            pl.when(pl.col("PROPERTY_TYPE") == "Flat")
            .then(pl.lit("Flat"))
            .when(pl.col("BUILT_FORM") == "Enclosed Mid-Terrace")
            .then(pl.lit("Mid-Terrace"))
            .when(pl.col("BUILT_FORM") == "Enclosed End-Terrace")
            .then(pl.lit("End-Terrace"))
            .when(pl.col("BUILT_FORM") == "")
            .then(pl.lit("unknown"))
            .otherwise(pl.col("BUILT_FORM"))
            .alias("attachment")
        )
    )

    print("Joining geospatial coordinates to EPC data...")
    # Add geospatial coordinates to EPC UPRNs
    epc_gdf = uprns_gdf.merge(
        epc_df.to_pandas(), how="inner", left_on="UPRN", right_on="processed_UPRN"
    )

    # Filter to UPRNs which are in opportunity areas
    areas_epc_gdf = epc_gdf.sjoin(
        areas_gdf[["Name", "geometry"]], how="inner", predicate="within"
    ).drop(columns="index_right")

    fname = "plymouth_example_opportunity_areas_feature_counts.txt"
    print(f"Saving outputs to {fname}")
    # Save the value counts of features of interest to txt file
    with open(fname, "w") as file:
        for tech in areas_epc_gdf["Name"].unique():
            _gdf = areas_epc_gdf[areas_epc_gdf["Name"] == tech].drop_duplicates(
                subset="processed_UPRN"
            )
            file.write(f"\n\n{tech}: total properties: {len(_gdf)}")

            file.write(f"\n\n{tech} tenure:")
            file.write(_gdf["TENURE"].value_counts().to_string())

            file.write(f"\n\n{tech} attachment:")
            file.write(_gdf["attachment"].value_counts().to_string())

            file.write(f"\n\n{tech} EPC rating:")
            file.write(
                _gdf["CURRENT_ENERGY_RATING"].value_counts().sort_index().to_string()
            )
