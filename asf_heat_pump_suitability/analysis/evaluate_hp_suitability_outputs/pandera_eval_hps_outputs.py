"""
This script performs data quality checks and statistical analysis on heat pump suitability data.

Key functionalities:
1. Validates the dataset schema using Pandera (Polars backend).
2. Detects outliers in numeric columns based on z-scores.
3. Generates summary statistics for numeric columns.
4. Logs results and saves a detailed numeric summary to a CSV file.
"""

from pathlib import Path
from datetime import datetime
import logging
import polars as pl
import pandera.polars as pa
import json
import config.dq_config as cfg
from pandera.polars import Check, Column, DataFrameSchema
from asf_heat_pump_suitability.analysis.hn_zones.hnz_utils.log_utils import (
    setup_logging_and_file_path,
)

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

setup_logging_and_file_path(output_dir=cfg.OUTPUT_DIR, log_filename=cfg.LOG_FILENAME)


# ---------------------------------------------------------------------------
# 1.  BUILD SCHEMA
# ---------------------------------------------------------------------------

schema_columns: dict[str, Column] = {}


def add(col: str, dtype, *checks, **kw):
    """
    Registers or updates a column in the `schema_columns` dictionary for schema validation.

    Args:
        col (str): The name of the column to register or update.
        dtype: The data type of the column (e.g., `float`, `str`, `pl.Boolean`).
        *checks: Any number of Pandera `Check` objects to validate column values.
        **kw: Additional keyword arguments for column properties (e.g., `required`, `nullable`, `unique`).
    """
    if col in schema_columns:  # merge with existing
        prev = schema_columns[col]
        checks = tuple(prev.checks) + tuple(checks)  # keep earlier checks
        kw = {**{k: getattr(prev, k) for k in ("required", "nullable", "unique")}, **kw}
    schema_columns[col] = Column(dtype, checks=list(checks), **kw)


# -- primary key ------------------------------------------------------------ #
add("lsoa", str, unique=True, required=True, nullable=False)

# -- score + proportion columns in [0, 1] ----------------------------------- #
for c in (*cfg.SCORE_COLUMNS, *cfg.PROPORTION_COLUMNS):
    add(c, float, Check.in_range(0, 1), required=True, nullable=True)

# -- boolean ---------------------------------------------------------------- #
for c in cfg.BOOLEAN_COLUMNS:
    add(c, pl.Boolean, required=True, nullable=True)

# -- categorical ------------------------------------------------------------ #
for c, allowed in cfg.CATEGORICAL_COLUMNS.items():
    add(c, str, Check.isin(allowed), required=True, nullable=True)

# -- non-negative ----------------------------------------------------------- #
for c in cfg.NON_NEGATIVE_COLUMNS:
    add(c, float, Check.ge(0), required=True, nullable=True)

# -- 0–100 percentage ------------------------------------------------------- #
add(
    "heatpump_installation_percentage",
    float,
    Check.in_range(0, 100),
    required=True,
    nullable=True,
)

# -- lsoa name -------------------------------------------------------------- #
add("lsoa_name", str, required=True, nullable=True)

# finally create the schema
schema = DataFrameSchema(schema_columns, strict=False, coerce=True)


# ------------------------------------------------------------------------------
# 2. LOAD WITH POLARS
# ------------------------------------------------------------------------------
try:
    df_pol = pl.read_csv(cfg.DATA_PATH)
    logging.info("Data loaded successfully with Polars!")
except FileNotFoundError:
    logging.error(f"File not found at path: {cfg.DATA_PATH}")
    raise
except Exception:
    logging.exception(f"Could not load data from {cfg.DATA_PATH}")
    raise

# ------------------------------------------------------------------------------
# 3. VALIDATE WITH PANDERA (Polars backend)
# ------------------------------------------------------------------------------
logging.info("=== Validating with Pandera Schema ===")
try:
    df_valid = schema.validate(df_pol, lazy=True)
    logging.info("Data validated successfully with Pandera!")
except pa.errors.SchemaErrors as err:
    logging.warning("Pandera validation failed!")
    logging.warning(json.dumps(err.message, indent=2))
    logging.info("Continuing with unvalidated data.")
    df_valid = df_pol.clone()


# ------------------------------------------------------------------------------
# 4. OUTLIER DETECTION (Z-Score) IN POLARS
# ------------------------------------------------------------------------------
# compute z-score columns for numeric columns
df_valid = df_valid.with_columns(
    [
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"{c}_z")
        for c in cfg.NUMERIC_COLUMNS
    ]
)
logging.info("=== Outlier Detection (Z-Score) ===")

Z = cfg.OUTLIER_ZSCORE_THRESHOLD
num_cols = cfg.NUMERIC_COLUMNS


# outlier counts (1-row wide → long)
outlier_counts = df_valid.select(
    [(pl.col(f"{c}_z") > Z).sum().alias(c) for c in num_cols]
).transpose(include_header=True, header_name="variable", column_names=["n_out"])

# non-null counts the same way
nonnull_counts = df_valid.select(
    [pl.col(c).is_not_null().sum().alias(c) for c in num_cols]
).transpose(include_header=True, header_name="variable", column_names=["n_total"])

outlier_counts = (
    outlier_counts.join(nonnull_counts, on="variable")
    .with_columns((pl.col("n_out") / pl.col("n_total") * 100).alias("pct"))
    .filter(pl.col("n_out") > 0)
)


if outlier_counts.is_empty():
    logging.info("No outliers detected (z > %.1f).", Z)
else:
    logging.warning("Columns with outliers (z > %.1f):", Z)
    for row in outlier_counts.iter_rows(named=True):
        logging.warning(
            "  - %s: %d rows (%.2f%%)", row["variable"], row["n_out"], row["pct"]
        )


# ------------------------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------------------------
logging.info("=== DATA QUALITY CHECK SUMMARY ===")
logging.info("Rows: %d, Columns: %d", df_valid.height, df_valid.width)
logging.info("Column dtypes: %s", df_valid.dtypes)

# --- Full numeric summary to CSV ------------------------------------------ #
summary_df = df_valid.select(cfg.NUMERIC_COLUMNS).describe()
summary_file = (
    Path(cfg.OUTPUT_DIR)
    / f"{Path(cfg.DATA_PATH).stem}_numeric_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
summary_df.write_csv(summary_file)
logging.info("Full numeric summary ➜ %s", summary_file)
logging.info("Data Quality Checks Complete.")
