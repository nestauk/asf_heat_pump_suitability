"""
Filter large datasets to just be for the Stoke ward. Saves on loading time during other scripts.

Data formatted:
- Anchor properties
- Greenspace
"""

import geopandas as gpd
from asf_heat_pump_suitability.research.exploratory.proof_of_concept_methodology.create_full_dataset import (
    stoke_getters,
)
from asf_heat_pump_suitability.pipeline.prepare_features import anchor_properties

if __name__ == "__main__":

    # Load the Stoke ward boundary
    stoke_ward_boundary = stoke_getters.load_stoke_bound().to_crs(epsg=4326)

    # Load, filter and save Stoke anchor properties
    anchor_properties_df = anchor_properties.load_gdf_and_process_poi().to_crs(
        epsg=4326
    )

    stoke_anchor_properties_df = gpd.sjoin(
        anchor_properties_df, stoke_ward_boundary, how="inner", predicate="intersects"
    )

    stoke_anchor_properties_df_formatted = stoke_anchor_properties_df[
        ["main_category", "geometry"]
    ].rename(columns={"main_category": "Anchor load type"})
    stoke_anchor_properties_df_formatted["Type"] = (
        "Anchor load"  # This is the name + colour in flourish
    )
    stoke_anchor_properties_df_formatted["Long"] = stoke_anchor_properties_df_formatted[
        "geometry"
    ].x
    stoke_anchor_properties_df_formatted["Lat"] = stoke_anchor_properties_df_formatted[
        "geometry"
    ].y

    stoke_anchor_properties_df_formatted.to_file(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_anchorloads.geojson",
        driver="GeoJSON",
    )

    # Load, filter and save Stoke greenspaces

    greenspace_df = stoke_getters.load_SX_Greenspace().to_crs(epsg=4326)

    stoke_greenspace_df = gpd.sjoin(
        greenspace_df,
        stoke_ward_boundary,
        how="inner",
        predicate="intersects",
    )

    stoke_greenspace_df_formatted = stoke_greenspace_df[
        ["function", "distName1", "geometry"]
    ].rename(columns={"function": "Greenspace type", "distName1": "Greenspace name"})

    stoke_greenspace_df_formatted.to_file(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_greenspace.geojson",
        driver="GeoJSON",
    )
