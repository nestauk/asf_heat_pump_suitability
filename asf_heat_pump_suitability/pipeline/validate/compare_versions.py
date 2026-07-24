"""
Compare two dated versions of a pipeline stage output for one local authority.

Reports row/UPRN count deltas, schema diff, UPRN churn, the tech-assignment
transition matrix (decision-tree stage), and a module-scoped commit log
between the two versions' recorded commits, read against the tolerance rubric
of the trigger that prompted the comparison. Writes a local markdown report
and logs a console summary.

Usage:
python -m asf_heat_pump_suitability.pipeline.validate.compare_versions \
    --stage decision_tree \
    --local_authority plymouth \
    --old_release_date 20260601 \
    --new_release_date 20260722 \
    --trigger methodology_change
"""

import polars as pl

from asf_heat_pump_suitability import config

UPRN_COL = "UPRN"
TECH_COL = "assigned_tech"

TOLERANCES = config["compare_versions"]["tolerances"]


def get_dict_tolerances(trigger: str) -> dict:
    """
    Get the tolerances for a comparison trigger's rubric from base.yaml.

    Args:
        trigger: why the comparison is being run, e.g. "methodology_change"

    Returns:
        dict: tolerance name to value

    Raises:
        KeyError: if no rubric is configured for the trigger
    """
    try:
        return TOLERANCES[trigger]
    except KeyError as error:
        raise KeyError(
            f"No tolerance rubric configured for trigger '{trigger}'; expected "
            f"one of {sorted(TOLERANCES)} in config['compare_versions']"
        ) from error


def generate_dict_count_delta(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict:
    """
    Compare row and distinct-UPRN counts between two versions of an output.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: old/new/delta counts; UPRN entries are None when either version
            has no UPRN column (e.g. cluster-level outputs)
    """
    counts = {
        "rows_old": df_old.height,
        "rows_new": df_new.height,
        "rows_delta": df_new.height - df_old.height,
        "uprns_old": None,
        "uprns_new": None,
        "uprns_delta": None,
    }
    if UPRN_COL in df_old.columns and UPRN_COL in df_new.columns:
        counts["uprns_old"] = df_old[UPRN_COL].n_unique()
        counts["uprns_new"] = df_new[UPRN_COL].n_unique()
        counts["uprns_delta"] = counts["uprns_new"] - counts["uprns_old"]
    return counts


def generate_dict_schema_diff(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict:
    """
    Diff column names and dtypes between two versions of an output.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: "added" and "removed" map column to dtype string;
            "dtype_changed" maps column to an (old, new) dtype string pair
    """
    schema_old = {col: str(dtype) for col, dtype in df_old.schema.items()}
    schema_new = {col: str(dtype) for col, dtype in df_new.schema.items()}
    return {
        "added": {col: schema_new[col] for col in schema_new if col not in schema_old},
        "removed": {
            col: schema_old[col] for col in schema_old if col not in schema_new
        },
        "dtype_changed": {
            col: (schema_old[col], schema_new[col])
            for col in schema_old
            if col in schema_new and schema_old[col] != schema_new[col]
        },
    }


def generate_dict_uprn_churn(df_old: pl.DataFrame, df_new: pl.DataFrame) -> dict | None:
    """
    Count UPRNs added, removed and retained between two versions of an output.

    UPRNs are compared as strings so a dtype change between versions does not
    read as full churn.

    Args:
        df_old: older version of the stage output
        df_new: newer version of the stage output

    Returns:
        dict: added/removed/retained counts and the removed share of old
            UPRNs, or None when either version has no UPRN column
    """
    if UPRN_COL not in df_old.columns or UPRN_COL not in df_new.columns:
        return None
    uprns_old = set(df_old[UPRN_COL].cast(pl.Utf8).to_list())
    uprns_new = set(df_new[UPRN_COL].cast(pl.Utf8).to_list())
    n_removed = len(uprns_old - uprns_new)
    return {
        "n_added": len(uprns_new - uprns_old),
        "n_removed": n_removed,
        "n_retained": len(uprns_old & uprns_new),
        "removed_share": n_removed / len(uprns_old) if uprns_old else 0.0,
    }


def generate_str_churn_note(churn: dict, max_removed_share: float) -> str | None:
    """
    Flag UPRN loss above the rubric's tolerance.

    Args:
        churn: churn counts, as returned by `generate_dict_uprn_churn`
        max_removed_share: rubric tolerance for the removed share of old UPRNs

    Returns:
        str: warning naming the observed share and the tolerance, or None
            when the churn is within tolerance
    """
    if churn["removed_share"] <= max_removed_share:
        return None
    return (
        f"WARNING: {churn['removed_share']:.1%} of old UPRNs were removed, above "
        f"the {max_removed_share:.1%} tolerance for this rubric."
    )


def generate_df_tech_transitions(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> pl.DataFrame:
    """
    Count tech-assignment transitions for UPRNs present in both versions.

    Args:
        df_old: older version of the UPRN-level decision-tree output
        df_new: newer version of the UPRN-level decision-tree output

    Returns:
        pl.DataFrame: one row per (assigned_tech_old, assigned_tech_new) pair
            with its UPRN count, largest first
    """
    old = df_old.select(
        pl.col(UPRN_COL).cast(pl.Utf8),
        pl.col(TECH_COL).alias("assigned_tech_old"),
    )
    new = df_new.select(
        pl.col(UPRN_COL).cast(pl.Utf8),
        pl.col(TECH_COL).alias("assigned_tech_new"),
    )
    return (
        old.join(new, on=UPRN_COL, how="inner")
        .group_by("assigned_tech_old", "assigned_tech_new")
        .agg(pl.len().alias("n_uprns"))
        .sort("n_uprns", "assigned_tech_old", "assigned_tech_new", descending=True)
    )
