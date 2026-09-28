# %%
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata
import geopandas as gpd
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.transform import outdoor_space
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# %%
def get_list_la_grid_squares(boundary_gdf, grid_gdf, buffer_m: float = 1000) -> list:
    """
    Return grid squares corresponding to a local authority or list of local authorities. The buffer (m) ensures that geographical features (e.g. buildings) straddling the LA boundary are captured if they fall into neighbouring grid squares.

    Args:
        local_authorities (list): Official ONS place name (e.g. "King's Lynn and West Norfolk") or a list of names (e.g. ["Glasgow", "Midlothian"]). Defaults None to return whole of GB.
        buffer_m (float): buffer distance around local authority boundary (default = 1000m).

    Returns:
        list: list of OS BNG grid squares corresponding to the input local authorities.
    """
    boundary = boundary_gdf.geometry.buffer(buffer_m).union_all()

    # clip grid square gdf to local authority boundaries and return grid squares
    grid_gdf = grid_gdf.clip(boundary)
    grid_squares = list(grid_gdf["bng_ref"])

    return grid_squares


# %%
def get_gdf_land_parcels(boundary_gdf):
    inspire_file_gdf = gpd.read_file(config["data"]["processed"]["inspire_file_names"])
    inspire_file_names = inspire_file_gdf.clip(boundary_gdf)[
        "inspire_file_name"
    ].unique()

    land_parcels_gdf = pd.concat(
        [
            outdoor_space.load_transform_gdf_land_parcels(f"s3://{file}")
            for file in inspire_file_names
        ],
        ignore_index=False,
    )
    land_parcels_gdf["geometry"] = land_parcels_gdf.normalize()
    parcels_gdf = land_parcels_gdf.drop_duplicates(subset=["geometry"])
    parcels_gdf = parcels_gdf.clip(boundary_gdf)
    return parcels_gdf


# %%
def get_gdf_missing_land_parcels(parcels_gdf, boundary_gdf):
    # Merge boundaries and parcels into single unified geometry footprints
    single_boundary_geom = boundary_gdf.union_all()
    all_parcels_geom = parcels_gdf.union_all()

    # Subtract the parcels from the local authority boundary
    missing_geom = single_boundary_geom.difference(all_parcels_geom)

    # Convert the result back into a GeoDataFrame
    missing_gdf = gpd.GeoDataFrame(geometry=[missing_geom], crs=parcels_gdf.crs)
    missing_gdf = missing_gdf.explode(index_parts=False).reset_index(drop=True)

    # Calculate the area for each missing gap
    missing_gdf["area"] = missing_gdf.geometry.area

    return missing_gdf


# %%
def get_gdf_missing_garden_data(parcels_gdf, grid_squares, boundary_gdf):
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )

    buildings_gdf = buildings_gdf.clip(boundary_gdf)

    # Get intersection of building footprint polygons and land polygons
    intersection_gdf = outdoor_space.generate_gdf_building_intersections(
        land_parcels_gdf=parcels_gdf,
        buildings_gdf=buildings_gdf,
    )

    # Get outdoor space
    outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
        building_intersections_gdf=intersection_gdf, land_parcels_gdf=parcels_gdf
    )

    garden_size = buildings_gdf.sjoin(
        outdoor_space_gdf, how="left", predicate="intersects"
    )

    missing_gardens = garden_size[garden_size["total_outdoor_space_area_m2"].isna()]

    return missing_gardens


# %%
def plot_spatial_distribution_land_parcels(
    boundary_gdf, missing_gdf, missing_gardens, local_authority
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the local authority boundary as the background
    boundary_gdf.plot(ax=ax, color="lightgrey", edgecolor="black")

    # Plot the missing areas, colored by their area
    missing_gdf.plot(
        ax=ax,
        column="area",
        cmap="Reds",
        legend=True,
        legend_kwds={"label": "Missing Area Size"},
        alpha=0.8,
    )

    # Plot missing garden data
    missing_gardens.plot(ax=ax)

    plt.title(f"Spatial Distribution of Missing Land Parcels within {local_authority}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
def plot_statistical_distribution_land_parcels(missing_gdf, local_authority):
    plt.figure(figsize=(10, 6))
    sns.histplot(missing_gdf["area"], bins=50, log_scale=True)
    plt.title(f"Distribution of Missing Area Sizes {local_authority}")
    plt.xlabel("Area (Square Units)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# %%
local_authorities = [
    "Plymouth",
    "Vale of Glamorgan",
    "Midlothian",
    "Dudley",
    "Glasgow City",
]

# %%
grid_gdf = load_geodata.load_gdf_bng_grid_squares()

# %%
for la in local_authorities:
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(la)
    grid_squares = get_list_la_grid_squares(
        boundary_gdf=boundary_gdf, grid_gdf=grid_gdf
    )

    parcels_gdf = get_gdf_land_parcels(boundary_gdf=boundary_gdf)
    missing_gdf = get_gdf_missing_land_parcels(
        parcels_gdf=parcels_gdf, boundary_gdf=boundary_gdf
    )
    missing_gardens = get_gdf_missing_garden_data(
        parcels_gdf=parcels_gdf, grid_squares=grid_squares, boundary_gdf=boundary_gdf
    )

    print(f"--- Missing Area Statistics for {la} ---")
    print(missing_gdf["area"].describe())

    total_missing = missing_gdf["area"].sum()
    total_parcel = parcels_gdf.geometry.area.sum()
    print(f"\nTotal Missing Area: {total_missing:.2f}")
    print(f"Total Parcel Area: {total_parcel:.2f}")

    plot_spatial_distribution_land_parcels(
        boundary_gdf=boundary_gdf,
        missing_gdf=missing_gdf,
        missing_gardens=missing_gardens,
        local_authority=la,
    )

    plot_statistical_distribution_land_parcels(
        missing_gdf=missing_gdf, local_authority=la
    )


# %%
def plot_garden_size_by_signature(
    df,
    la_name,
    target_col="max_contiguous_outdoor_space_area_m2",
    sig_col="spatial_signature_types",
):
    plt.figure(figsize=(12, 6))

    # Drop NaNs just for the plot
    plot_data = df.dropna(subset=[target_col, sig_col])

    sns.boxplot(
        data=plot_data,
        x=sig_col,
        y=target_col,
        palette="viridis",  # Uses a nice color gradient
    )

    # Set the Y-axis to a logarithmic scale
    plt.yscale("log")

    plt.title(
        f"Distribution of Garden Sizes by Spatial Signature {la_name}",
        fontsize=16,
        pad=15,
    )
    plt.xlabel("Spatial Signature Type", fontsize=12)
    plt.ylabel("Garden Size (m²) [Log Scale]", fontsize=12)

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{la_name}_garden_size_by_spatial_sig")
    plt.show()


# %%
local_authority_slugs = [
    "plymouth",
    "vale_of_glamorgan",
    "midlothian",
    "dudley",
    "glasgow",
]

# %%
for la in local_authority_slugs:
    uprn_with_features = pd.read_parquet(
        f"s3://asf-local-heat-planning-tool/outputs/data/{la}/{la}_with_features.parquet"
    )
    uprn_with_features["spatial_signature_types"] = uprn_with_features[
        "spatial_signature_types"
    ].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) > 0 else x)
    print(
        f"percentage of uprns with no outdoor space data {la}: {sum(uprn_with_features["max_contiguous_outdoor_space_area_m2"].isna())/len(uprn_with_features)*100}%"
    )
    plot_garden_size_by_signature(uprn_with_features, la_name=la)

# %%


# %%
