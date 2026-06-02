import geopandas as gpd
import logging
import pandas as pd
import polars as pl
import difflib
import regex as re
from asf_heat_pump_suitability.getters import load_geodata, load_boundaries
from asf_heat_pump_suitability.utils import geo_utils
from asf_heat_pump_suitability import config


def resolve_list_la_names(la_names: str | list[str]) -> list:
    """
    Check input local authority names are valid (not case sensitive), and suggest a similar LA name if not.

    Args:
        la_names (str | list[str]): local authority names

    Returns:
        list: list of valid local authority names, in the format found in the local authority boundary data.

    Raises:
        ValueError: If any provided local authority name is not found in the valid list.
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
                    raise ValueError(f"No match found for: '{la_name}'")
                else:
                    suggestions = [possible_la_names_map[m] for m in matches]
                    raise ValueError(
                        f"{la_name} not found. Did you mean: {', '.join(suggestions)}?"
                    )

    return resolved_list


def make_str_slug(name: str | list[str]) -> str:
    """
    Convert an official ONS place name to a URL-safe hyphenated slug for saving data.

    Rules applied in order:
    1. Lowercase the name.
    2. Strip apostrophes and other single-quote characters.
    3. Replace any run of non-alphanumeric characters with a single hyphen.
    4. Strip leading/trailing hyphens.

    Args:
        name (str | list[str]): string e.g. an official ONS place name ("King's Lynn and West Norfolk") or a list of strings (e.g. ["Glasgow", "Midlothian"]).

    Returns:
        str: URL-safe slug (e.g. "kings_lynn_and_west_norfolk" or "glasgow_midlothian")
    """
    # Convert a single string to a 1-element list so the rest of the logic is identical
    names = [name] if isinstance(name, str) else name

    slug_parts = []
    for item in names:
        item = str(item).lower()
        item = re.sub(r"['\u2018\u2019\u02bc]", "", item)
        item = re.sub(r"[^a-z0-9]+", "_", item)
        cleaned = item.strip("_")
        if cleaned:  # Avoid adding empty strings to the final slug
            slug_parts.append(cleaned)

    return "-".join(slug_parts)


def get_list_la_grid_squares(
    local_authorities: list = None, buffer_m: float = 1000
) -> list:
    """
    Return grid squares corresponding to a local authority or list of local authorities. The buffer (m) ensures that geographical features (e.g. buildings) straddling the LA boundary are captured if they fall into neighbouring grid squares.

    Args:
        local_authorities (list): Official ONS place name (e.g. "King's Lynn and West Norfolk") or a list of names (e.g. ["Glasgow", "Midlothian"]). Defaults None to return whole of GB.
        buffer_m (float): buffer distance around local authority boundary (default = 1000m).

    Returns:
        list: list of OS BNG grid squares corresponding to the input local authorities.
    """
    # Get all BNG grid squares
    grid_gdf = load_geodata.load_gdf_bng_grid_squares()

    # Total boundary of all local authorities of interest, with a buffer of buffer_m
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        local_authorities
    )
    boundary_gdf = boundary_gdf.geometry.buffer(buffer_m).union_all()

    # clip grid square gdf to local authority boundaries and return grid squares
    grid_gdf = grid_gdf.clip(boundary_gdf)
    grid_squares = set(grid_gdf["bng_ref"])

    return grid_squares


def get_dict_la_data(la_names: str | list[str]) -> dict:
    """
    Takes a local authority or list of local authority names and returns a dictionary
    with a single combined URL-slug as the key, and the resolved LAs
    and grid squares as the values.

    Args:
        la_names (str | list[str]): List of local authority names.

    Returns:
        dict: Format -> { "url-slug": str, "valid_local_authorities": [...], "grid_squares": [...] }
    """
    resolved_las = resolve_list_la_names(la_names)
    # whole of GB
    if resolved_las is None:
        slug = "gb"
        grid_squares = load_geodata.load_gdf_bng_grid_squares()
        grid_squares_set = set(grid_squares["bng_ref"])
    else:
        # Generate the URL slug from the official resolved names
        slug = make_str_slug(resolved_las)
        # Fetch the combined grid squares
        grid_squares_set = get_list_la_grid_squares(resolved_las)

    # Construct the output dictionary
    return {
        "url_slug": slug,
        "valid_local_authorities": resolved_las,
        "grid_squares": list(grid_squares_set),
    }
