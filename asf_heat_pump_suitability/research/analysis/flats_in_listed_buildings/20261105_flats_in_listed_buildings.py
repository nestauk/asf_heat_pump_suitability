# %% [markdown]
# ## Calculate proportion of listed buildings that are flats
#
# This notebook calculates the total number of listed buildings containing domestic properties in GB. It calculates:
# 1. The number and proportion of listed buildings which are classified as blocks of flats by our model.
# 2. The number and proportion of listed buildings in which >80% are flats according to our predictions.
#
# Caveats and assumptions:
# - The count of buildings may be underestimated as the building footprints used in this analysis sometimes contain multiple buildings within a single footprint.
# - The listed buildings geospatial datasets used to identify listed buildings may not be complete.
# - The methodology used here for identifying listed buildings may result in neighbouring buildings being mislabelled as listed in some cases.
# - The model to classify building footprints as a block of flats or not is primarily designed for application of the LHPT's decision tree. For this use case we define a block of flats as: "a residential building with 3 or more storeys divided into several separate individual properties (e.g. tower blocks, tenements, 3 or more storeys of flats above shops and large mansions converted into 3 or more storeys of flats)." This means that small blocks of flats (e.g. 2-storey social housing blocks) will not be included in this definition. Find more information in section 2.b of the methodology README https://media.nesta.org.uk/documents/full_methodology_README.pdf .
# - Flats may be located within purpose-built blocks within converted houses or other buildings. A UPRN is considered to represent a flat if it is labelled by our pipeline as domestic and is located at the exact same coordinates as any other domestic UPRN. This is based on the assumption that UPRNs in the exact same location share an address point (e.g. a front door) and so are most likely to represent flats.

# %%
from tqdm import tqdm
import polars as pl
import geopandas as gpd
import pandas as pd
from asf_heat_pump_suitability.pipeline.transform import local_authority

# %%
date = "20261030"

nations = {"E": "England", "S": "Scotland", "W": "Wales"}

results = []

for nation_code, nation_name in nations.items():
    print(f"\n\nProcessing LAs for {nation_name}")
    # Load LAs for the nation
    la_names = local_authority.load_list_la_names(nation=nation_code)
    # Ensure names are consistent with file names
    resolved_las = local_authority.resolve_list_la_names(la_names)
    for la in tqdm(resolved_las):
        slug = local_authority.make_str_slug(la)
        # Load UPRN data enriched with features
        uprns_gdf = gpd.read_parquet(
            f"s3://asf-local-heat-planning-tool/outputs/data/{slug}/{date}/{slug}_uprns_most_suitable_tech.parquet"
        )
        # Filter to those located in listed buildings and aggregate to building level
        listed_uprns_gdf = uprns_gdf[uprns_gdf["in_listed_building"]]
        listed_building_df = listed_uprns_gdf.groupby("ID").agg(
            {"in_block_of_flats": "first", "property_type_flat": "mean"}
        )
        # Add column where the proportion of UPRNs which are flats is over 80% (boolean)
        listed_building_df["gt_80_pc_flats"] = (
            listed_building_df["property_type_flat"] > 0.8
        )
        listed_building_df[["local_authority", "nation"]] = slug, nation_name
        results.append(listed_building_df)

results_df = pl.from_pandas(pd.concat(results))

# %%
all_listed_buildings = results_df.height
n_listed_blocks_of_flats = results_df["in_block_of_flats"].sum()
n_listed_80pc_flats = results_df["gt_80_pc_flats"].sum()

print(
    f"Total number of listed buildings containing domestic properties in GB: {all_listed_buildings}\n"
    f"Number of listed buildings which are blocks of flats: {n_listed_blocks_of_flats}; percentage: {round(n_listed_blocks_of_flats / all_listed_buildings * 100, 2)}%\n"
)

print(
    f"Number of listed buildings where >80% of properties are flats: {n_listed_80pc_flats}; percentage: {round(n_listed_80pc_flats / all_listed_buildings * 100, 2)}%\n"
)
