import geopandas as gpd
import logging
import pandas as pd
import polars as pl
import difflib
import regex as re
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries
from asf_heat_pump_suitability.utils import geo_utils
from asf_heat_pump_suitability import config


def resolve_la_names(la_names: str | list[str]) -> list:
    """
    Check input local authority names are valid (not case sensitive), and suggest a similar LA name if not.

    Args:
        la_names: local authority names

    Returns:
        list of valid local authority names, in the format found in the local authority boundary data.
    """
    # Load reference data of possible LA names
    possible_la_names = pd.read_csv(config["data"]["processed"]["valid_la_names"])
    possible_la_names = possible_la_names["LAD23NM"]

    # Map for lookup
    possible_la_names_map = {name.lower(): name for name in possible_la_names}

    # Ensure we are working with a list
    if isinstance(la_names, str):
        la_names = [la_names]

    # If running for whole of GB, return no LA names
    if la_names is None or "gb" in [name.lower() for name in la_names]:
        resolved_list = None

    else:

        resolved_list = []

        for la_name in la_names:
            target = la_name.lower().strip()

            if target in possible_la_names_map:
                resolved_list.append(possible_la_names_map[target])
            else:
                matches = difflib.get_close_matches(
                    target, possible_la_names_map.keys(), n=3, cutoff=0.7
                )

                if not matches:
                    print(f"No match found for: '{la_name}'")
                    resolved_list.append(None)
                else:
                    suggestions = [possible_la_names_map[m] for m in matches]
                    print(
                        f"{la_name} not found. Did you mean: {', '.join(suggestions)}?"
                    )
                    resolved_list.append(None)

    return resolved_list


def make_slug(name: str | list[str]) -> str:
    """
    Convert an official ONS place name to a URL-safe hyphenated slug for saving data.

    Rules applied in order:
    1. Lowercase the name.
    2. Strip apostrophes and other single-quote characters.
    3. Replace any run of non-alphanumeric characters with a single hyphen.
    4. Strip leading/trailing hyphens.

    Args:
        name: Official ONS place name (e.g. "King's Lynn and West Norfolk") or a list of names (e.g. ["Glasgow", "Midlothian"]).

    Returns:
        URL-safe slug (e.g. "kings-lynn-and-west-norfolk" or "glasgow-midlothian")
    """
    # Convert a single string to a 1-element list so the rest of the logic is identical
    names = [name] if isinstance(name, str) else name

    slug_parts = []
    for item in names:
        item = str(item).lower()
        item = re.sub(r"['\u2018\u2019\u02bc]", "", item)
        item = re.sub(r"[^a-z0-9]+", "-", item)
        cleaned = item.strip("-")
        if cleaned:  # Avoid adding empty strings to the final slug
            slug_parts.append(cleaned)

    return "-".join(slug_parts)


def get_la_grid_squares(local_authorities: list = None, buffer_m: float = 1000) -> list:
    """
    Return grid squares corresponding to a local authority or list of local authorities. Defaults None to return whole of GB.

    Args:
        local_authorities: Official ONS place name (e.g. "King's Lynn and West Norfolk") or a list of names (e.g. ["Glasgow", "Midlothian"]).
        buffer_m: buffer distance around local authority boundary (default = 1000m).

    Returns:
        list of OS BNG grid squares corresponding to the input local authorities.
    """

    # For whole of GB, return None
    if local_authorities is None:
        grid_squares = None

    else:
        # Get all BNG grid squares
        grid_gdf = load_geodata.load_gdf_bng_grid_squares()

        # Total boundary of all local authorities of interest, with a buffer of buffer_m
        boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            local_authorities
        )
        boundary_gdf = boundary_gdf.geometry.buffer(buffer_m).union_all()

        # clip grid square gdf to local authority boundaries and return grid squares
        grid_gdf = grid_gdf.clip(boundary_gdf)
        grid_squares = list(grid_gdf["bng_ref"].drop_duplicates())

    return grid_squares
