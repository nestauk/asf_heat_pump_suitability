import geopandas as gpd
import pandas as pd
import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import (
    base_getters,
    load_tree_input,
    load_boundaries,
)
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.utils import mapping_utils


def transform_gdf_council_tax(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert council tax UPRN data into geodataframe with point geometries per UPRN. Note, this drops rows without UPRN
    coordinates.

    Args:
        df (pd.DataFrame): raw council tax UPRN data

    Returns:
        gpd.GeoDataFrame: council tax UPRNs with point geometries
    """
    # Remove empty UPRN and coordinate rows
    df = df[(df["UPRN"] != "") & (df["EASTING"] != "") & (df["NORTHING"] != "")]

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["EASTING"], df["NORTHING"]),
        crs="EPSG:27700",
    ).drop_duplicates(subset="UPRN")


def calculate_dict_uprn_diffs_per_dataset(
    raw_council_tax_df, council_tax_gdf, pipeline_gdf
):
    council_uprns = set(council_tax_gdf["UPRN"])
    pipeline_uprns = set(pipeline_gdf["UPRN"])

    n_council_uprns = council_tax_gdf["UPRN"].nunique()
    n_pipeline_uprns = pipeline_gdf["UPRN"].nunique()

    results = {
        # Number of unique records / UPRNs per dataset
        "N unique records in council tax data": raw_council_tax_df["PROPREF"].nunique(),
        "N unique UPRNs in council tax data": n_council_uprns,
        "N unique UPRNs in pipeline output data": n_pipeline_uprns,
        # Differences in council tax UPRNs versus pipeline
        "N diff UPRNs in pipeline minus council tax": n_pipeline_uprns
        - n_council_uprns,
        "Proportion diff UPRNs in pipeline minus council tax": round(
            (n_pipeline_uprns - n_council_uprns) / n_pipeline_uprns, 3
        ),
        "N domestic UPRNs missing from pipeline": len(
            council_uprns.difference(pipeline_uprns)
        ),
        "Proportion domestic UPRNs missing from pipeline": round(
            len(council_uprns.difference(pipeline_uprns)) / n_council_uprns, 3
        ),
        "N UPRNs in pipeline but not council tax": len(
            pipeline_uprns.difference(council_uprns)
        ),
        "Proportion UPRNs in pipeline not in council tax": round(
            len(pipeline_uprns.difference(council_uprns)) / n_pipeline_uprns, 3
        ),
    }

    return results


def calculate_dict_building_diffs_per_dataset(
    n_council_records: int,
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
):
    council_buildings_gdf = buildings_gdf.sjoin(
        council_tax_gdf, how="inner", predicate="within"
    ).drop(columns="index_right")
    pipeline_buildings_gdf = buildings_gdf.sjoin(
        pipeline_gdf, how="inner", predicate="within"
    ).drop(columns="index_right")

    council_buildings = set(council_buildings_gdf["ID"])
    pipeline_buildings = set(pipeline_buildings_gdf["ID"])

    n_council_buildings = council_buildings_gdf["ID"].nunique()
    n_pipeline_buildings = pipeline_buildings_gdf["ID"].nunique()

    results = {
        "Proportion of council records located in building footprints": round(
            council_buildings_gdf["UPRN"].nunique() / n_council_records, 3
        ),
        "Proportion of council UPRNs located in building footprints": round(
            council_buildings_gdf["UPRN"].nunique() / len(council_tax_gdf), 3
        ),
        "Proportion of pipeline UPRNs located in building footprints": round(
            pipeline_buildings_gdf["UPRN"].nunique() / len(pipeline_gdf), 3
        ),
        "Proportion of council buildings in pipeline buildings": round(
            len(council_buildings.intersection(pipeline_buildings))
            / n_council_buildings,
            3,
        ),
        "N pipeline buildings containing a council tax UPRN": len(
            pipeline_buildings.intersection(council_buildings)
        ),
        "Proportion of pipeline buildings containing a council tax UPRN": round(
            len(pipeline_buildings.intersection(council_buildings))
            / n_pipeline_buildings,
            3,
        ),
    }

    return results


def generate_gdf_erroneous_pipeline_buildings(
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
):
    council_buildings_gdf = buildings_gdf.sjoin(
        council_tax_gdf, how="inner", predicate="within"
    ).drop(columns="index_right")
    pipeline_buildings_gdf = buildings_gdf.sjoin(
        pipeline_gdf, how="inner", predicate="within"
    ).drop(columns="index_right")

    council_buildings = set(council_buildings_gdf["ID"])

    false_positives_gdf = pipeline_buildings_gdf[
        ~pipeline_buildings_gdf["ID"].isin(council_buildings)
    ]
    uprns_per_building_gdf = false_positives_gdf.groupby("ID").agg(
        UPRN_count=("UPRN", "nunique"), geometry=("geometry", "first")
    )

    return gpd.GeoDataFrame(
        uprns_per_building_gdf, geometry="geometry", crs="EPSG:27700"
    )


if __name__ == "__main__":

    # LOAD DATASETS FOR ANALYSIS
    raw_council_tax_uprns_gdf = pd.read_csv(
        config["data"]["geodata"]["council_tax_data"]["plymouth"]
    )
    pipeline_domestic_uprns_df = base_getters.load_df_from_s3(
        config["data"]["processed"]["plymouth_residential_uprns"]
    )
    pipeline_domestic_uprns_gdf = uprns.generate_gdf_uprn_coords(
        pipeline_domestic_uprns_df
    )

    buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares="SX"
    )
    plymouth_boundary = load_boundaries.load_gdf_local_authority_boundaries(
        select_las="Plymouth"
    )["geometry"].values[0]

    n_council_records = len(raw_council_tax_uprns_gdf)
    council_tax_uprns_gdf = transform_gdf_council_tax(raw_council_tax_uprns_gdf)

    results = calculate_dict_uprn_diffs_per_dataset(
        raw_council_tax_uprns_gdf, council_tax_uprns_gdf, pipeline_domestic_uprns_gdf
    )
    buildings_results = calculate_dict_building_diffs_per_dataset(
        n_council_records=n_council_records,
        council_tax_gdf=council_tax_uprns_gdf,
        pipeline_gdf=pipeline_domestic_uprns_gdf,
        buildings_gdf=buildings_gdf,
    )

    false_positives_gdf = generate_gdf_erroneous_pipeline_buildings(
        council_tax_gdf=council_tax_uprns_gdf,
        pipeline_gdf=pipeline_domestic_uprns_gdf,
        buildings_gdf=buildings_gdf,
    )
    mapping_utils.plot_folium_polygon_map(
        polygon_gdf=false_positives_gdf,
        boundary=plymouth_boundary,
        popup_col="UPRN_count",
        save_as="plymouth_false_positive_domestic_buildings",
    )
