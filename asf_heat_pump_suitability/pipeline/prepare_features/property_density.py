import polars as pl
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import household_count


def generate_df_property_density() -> pl.DataFrame:
    """
    Generate dataframe with property density (households per km2) per LSOA/2011 Data Zone in England and Wales, and Scotland,
    respectively, using Standard Area Measurement of ‘Land Area’ (Area to Mean High Water Excluding Area of Inland Water).

    Returns:
        pl.DataFrame: households per km2 per LSOA/Data Zone in England, Wales, and Scotland
    """
    ew_df = generate_df_property_density_ew()
    s_df = generate_df_property_density_s().rename({"DataZone": "lsoa"})

    return pl.concat([ew_df, s_df])


def generate_df_property_density_ew() -> pl.DataFrame:
    """
    Generate dataframe with property density (households per km2) per LSOA in England and Wales using Standard Area Measurement of
    ‘Land Area’ (Area to Mean High Water Excluding Area of Inland Water).

    Returns:
        pl.DataFrame: households per km2 per LSOA in England and Wales
    """
    households_df = household_count.load_transform_df_n_households_ew()
    area_df = load_transform_df_land_area_ew()
    df = households_df.join(area_df, how="inner", on="lsoa")
    df = df.with_columns(
        (pl.col("households_count") / pl.col("Land Count (Area in KM2)")).alias(
            "households_per_km2"
        )
    )

    return df.select(["lsoa", "households_per_km2"])


def load_transform_df_land_area_ew() -> pl.DataFrame:
    """
    Process and clean ONS land area dataset for England and Wales.

    Returns
        pl.DataFrame: processed ONS land area
    """
    df = get_datasets.get_df_ons_land_area().rename(
        {
            "LSOA21CD": "lsoa",
        }
    )

    return df.select(["lsoa", "Land Count (Area in KM2)"])


def generate_df_property_density_s() -> pl.DataFrame:
    """
    Generate dataframe with property density (dwellings per km2) per 2011 Data Zone in Scotland. Long-term empty dwellings
    are excluded from dwelling count to calculate property density.

    Returns:
        pl.DataFrame: properties per km2 per 2011 Data Zone in Scotland
    """
    dz_df = load_transform_df_datazone_area()
    dwellings_df = household_count.load_transform_df_n_dwellings_s()
    df = dwellings_df.join(dz_df, how="inner", on="DataZone").with_columns(
        (pl.col("n_dwellings") / pl.col("StdAreaKm2")).alias("households_per_km2")
    )

    return df.select(["DataZone", "households_per_km2"])


def load_transform_df_datazone_area() -> pl.DataFrame:
    """
    Load and transform dataframe with area (km2) per 2011 Scottish Data Zone. Uses Standard Area Measurement of
    ‘Land Area’ (Area to Mean High Water Excluding Area of Inland Water).

    Returns:
         pl.DataFrame: area (km2) per 2011 Data Zone in Scotland
    """
    df = get_datasets.load_gdf_scotgov_data_zone_bounds()[["DataZone", "StdAreaKm2"]]

    return pl.from_pandas(df)
