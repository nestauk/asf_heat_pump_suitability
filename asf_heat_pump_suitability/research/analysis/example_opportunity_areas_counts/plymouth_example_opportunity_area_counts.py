"""
Script to use EPC data to get value counts of TENURE, ATTACHMENT TYPE, and EPC RATING for
3 example opportunity areas in Plymouth.

To run:
python asf_heat_pump_suitability/research/analysis/example_opportunity_areas_counts/plymouth_example_opportunity_area_counts.py
"""

if __name__ == "__main__":
    import polars as pl
    import geopandas as gpd
    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.getters import load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns

    # Load UPRNs
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    # Load EPC
    print("Loading deduplicated EPC data...")
    raw_epc_df = pl.read_parquet(
        "s3://asf-daps/lakehouse/2025_Q1/processed/epc/deduplicated/processed_dedupl-0.parquet"
    )

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
            .alias("UPRN")
        )
        .drop_nulls(subset="UPRN")
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
            .when(pl.col("BUILT_FORM").str.to_lowercase().is_in(["", "unknown"]))
            .then(None)
            .otherwise(pl.col("BUILT_FORM"))
            .alias("ATTACHMENT")
        )
        .with_columns(
            pl.col("TENURE")
            .str.to_lowercase()
            .replace({"": None, "unknown": None})
            .alias("TENURE")
        )
    )

    print("Joining EPC data to UPRNs...")
    # Add EPC data to UPRNs
    keep_cols = ["UPRN", "ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    uprns_gdf = uprns_gdf.merge(epc_df.to_pandas(), how="left", on="UPRN").fillna(
        "Unknown"
    )

    # Filter to UPRNs which are in opportunity areas
    opportunity_areas_df = pl.from_pandas(
        uprns_gdf.sjoin(
            areas_gdf[["Name", "geometry"]], how="right", predicate="within"
        ).drop(columns=["index_left", "geometry"])
    )

    print("Calculate value counts per feature...")
    keep_cols += ["Name"]
    dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    totals_df = opportunity_areas_df.group_by("Name").agg(
        pl.col("UPRN").count().alias("n_UPRNs")
    )
    opportunity_areas_df = (
        opportunity_areas_df.select(keep_cols)
        .to_dummies(columns=dummy_cols)
        .group_by("Name")
        .agg(pl.all().sum())
        .drop("UPRN")
    )

    opportunity_areas_df = totals_df.join(
        opportunity_areas_df, how="left", on="Name"
    ).rename({"Name": "opportunity_area_code"})

    path = "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth/plymouth_example_opportunity_areas_feature_counts.csv"
    save_utils.save_to_s3(opportunity_areas_df, path)
