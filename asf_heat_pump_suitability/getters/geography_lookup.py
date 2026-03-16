"""ONS geography lookup: slug generation, lookup loading, and LAD code resolution."""

import re

import pandas as pd

from asf_heat_pump_suitability import config


def make_slug(name: str) -> str:
    """Convert an official ONS place name to a URL-safe hyphenated slug.

    Rules applied in order:
    1. Lowercase the name.
    2. Strip apostrophes and other single-quote characters.
    3. Replace any run of non-alphanumeric characters with a single hyphen.
    4. Strip leading/trailing hyphens.

    Args:
        name: Official ONS place name (e.g. "King's Lynn and West Norfolk").

    Returns:
        URL-safe slug (e.g. "kings-lynn-and-west-norfolk").
    """
    name = name.lower()
    name = re.sub(r"['\u2018\u2019\u02bc]", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def get_geography_lookup() -> pd.DataFrame:
    """Read the pre-built LAD->UTLA->combined-authority lookup from S3.

    Returns one row per LAD with name, code, and slug columns for all three tiers.
    Built by ``asf_heat_pump_suitability/pipeline/setup/build_geography_lookup.py``.

    Returns:
        DataFrame with columns: lad_code, lad_name, lad_slug, utla_code, utla_name,
        utla_slug, combined_code, combined_name, combined_slug.
    """
    path = config["inputs"]["reference"]["geography_lookup"]
    return pd.read_parquet(path)


def resolve_lads(
    lads: list[str] | None,
    utlas: list[str] | None,
    combineds: list[str] | None,
    lookup: pd.DataFrame,
) -> list[str]:
    """Resolve CLI geography arguments to a list of LAD codes.

    Each argument may be an official name, a slug, or an ONS code. Matching is
    case-insensitive on both name and slug columns. Returns deduplicated, sorted
    LAD codes.

    If all arguments are None/empty, returns all LAD codes in the lookup.

    Args:
        lads: List of LAD names/slugs/codes, or None.
        utlas: List of UTLA names/slugs/codes, or None.
        combineds: List of combined authority names/slugs/codes, or None.
        lookup: Geography lookup DataFrame (from ``get_geography_lookup()``).

    Returns:
        Sorted, deduplicated list of LAD23CD codes.

    Raises:
        ValueError: If any supplied name/slug/code is not found in the lookup.
    """
    if not any([lads, utlas, combineds]):
        return sorted(lookup["lad_code"].tolist())

    codes: set[str] = set()

    if lads:
        for val in lads:
            v = val.lower()
            mask = (
                lookup["lad_code"].str.lower().eq(v)
                | lookup["lad_name"].str.lower().eq(v)
                | lookup["lad_slug"].str.lower().eq(v)
            )
            matched = lookup.loc[mask, "lad_code"].tolist()
            if not matched:
                raise ValueError(f"LAD not found: {val!r}")
            codes.update(matched)

    if utlas:
        for val in utlas:
            v = val.lower()
            mask = (
                lookup["utla_code"].str.lower().eq(v)
                | lookup["utla_name"].str.lower().eq(v)
                | lookup["utla_slug"].str.lower().eq(v)
            )
            matched = lookup.loc[mask, "lad_code"].tolist()
            if not matched:
                raise ValueError(f"UTLA not found: {val!r}")
            codes.update(matched)

    if combineds:
        for val in combineds:
            v = val.lower()
            # combined_* columns are nullable; fillna("") avoids NaN propagation
            c_code = lookup["combined_code"].fillna("").str.lower()
            c_name = lookup["combined_name"].fillna("").str.lower()
            c_slug = lookup["combined_slug"].fillna("").str.lower()
            mask = c_code.eq(v) | c_name.eq(v) | c_slug.eq(v)
            matched = lookup.loc[mask, "lad_code"].tolist()
            if not matched:
                raise ValueError(f"Combined authority not found: {val!r}")
            codes.update(matched)

    return sorted(codes)
