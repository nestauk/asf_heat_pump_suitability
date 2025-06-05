"""
This script performs data quality checks and statistical analysis on heat pump suitability data. Specifically this takes the heat pump suitability per LSOA data which is an output from asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_suitability.py

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
import argparse
from pandera.polars import Check, Column, DataFrameSchema
from asf_heat_pump_suitability.analysis.hn_zones.hnz_utils.log_utils import (
    setup_logging_and_file_path,
)
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
from asf_heat_pump_suitability import PROJECT_DIR
from typing import Tuple

# ------------------------------------------------------------------------------
# Argument parsing for data path
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="data quality checks for heat pump suitability data."
)
parser.add_argument(
    "--data-path",
    type=str,
    default=cfg.DATA_S3_URI,
    help="Path to the heat pump suitability per LSOA data CSV file (local or S3 URI).",
)
args = parser.parse_args()
data_path = args.data_path


# ------------------------------------------------------------------------------
# Util functions
# ------------------------------------------------------------------------------
def make_paths_from_data_path(data_path: str, project_dir: str) -> Tuple[str, str]:
    """Creates a timestamped logs directory and log filename from a data path.

    Args:
        data_path (str): Local file path or S3 URI of the data file.
        project_dir (str): Root directory of the project.

    Returns:
        Tuple[str, str]:
            - output_dir: Path to the created logs directory.
            - logfile: Log filename in the format "<base>_<TIMESTAMP>_dq.log".
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if data_path.startswith("s3://"):
        key = urlparse(data_path).path.lstrip("/")
    else:
        key = Path(data_path).name
    base = Path(key).stem
    outdir = (
        Path(project_dir) / "analysis" / "evaluate_hp_suitability_outputs" / "logs" / ts
    )
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = f"{base}_{ts}_dq.log"
    return str(outdir), logfile


output_dir, log_filename = make_paths_from_data_path(data_path, PROJECT_DIR)

setup_logging_and_file_path(output_dir=output_dir, log_filename=log_filename)


# ---------------------------------------------------------------------------
# 1.  BUILD SCHEMA
# ---------------------------------------------------------------------------


def add_schema_column(
    schema_columns: dict[str, Column], col: str, dtype: any, *checks, **kw
):
    """
    Registers or updates a column in the `schema_columns` dictionary for schema validation.

    Args:
        schema_columns (dict): The dictionary holding schema columns.
        col (str): The name of the column to register or update.
        dtype: The data type of the column (e.g., `float`, `str`, `pl.Boolean`).
        *checks: Any number of Pandera `Check` objects to validate column values.
        **kw: Additional Pandera keyword arguments for column properties (e.g., `required`, `nullable`, `unique`).
    """
    if col in schema_columns:  # merge with existing
        prev = schema_columns[col]
        checks = tuple(prev.checks) + tuple(checks)  # keep earlier checks
        kw = {**{k: getattr(prev, k) for k in ("required", "nullable", "unique")}, **kw}
    schema_columns[col] = Column(dtype, checks=list(checks), **kw)


schema_col = {}
# -- primary key ------------------------------------------------------------ #
add_schema_column(
    schema_col, col="lsoa", dtype=str, unique=True, required=True, nullable=False
)

# -- score + proportion columns in [0, 1] ----------------------------------- #
for c in (*cfg.SCORE_COLUMNS, *cfg.PROPORTION_COLUMNS):
    add_schema_column(
        schema_col, c, float, Check.in_range(0, 1), required=True, nullable=True
    )

# -- boolean ---------------------------------------------------------------- #
for c in cfg.BOOLEAN_COLUMNS:
    add_schema_column(schema_col, col=c, dtype=pl.Boolean, required=True, nullable=True)

# -- categorical ------------------------------------------------------------ #
for c, allowed in cfg.CATEGORICAL_COLUMNS.items():
    add_schema_column(
        schema_col, c, str, Check.isin(allowed), required=True, nullable=True
    )

# -- non-negative ----------------------------------------------------------- #
for c in cfg.NUMERIC_COLUMNS:
    add_schema_column(schema_col, c, float, Check.ge(0), required=True, nullable=True)

# -- 0–100 percentage ------------------------------------------------------- #
add_schema_column(
    schema_col,
    "heatpump_installation_percentage",
    float,
    Check.in_range(0, 100),
    required=True,
    nullable=True,
)

# -- lsoa name -------------------------------------------------------------- #
add_schema_column(schema_col, "lsoa_name", str, required=True, nullable=True)

# finally create the schema
schema = DataFrameSchema(schema_col, strict=False, coerce=True)


# ------------------------------------------------------------------------------
# 2. LOAD WITH POLARS
# ------------------------------------------------------------------------------
try:
    df_pol = pl.read_csv(data_path)
    logging.info("Data loaded successfully with Polars!")
except FileNotFoundError:
    logging.error(f"File not found at path: {data_path}")
    raise
except Exception:
    logging.exception(f"Could not load data from {data_path}")
    raise

# log any columns in the data that aren’t covered by the schema
extra_cols = set(df_pol.columns) - set(schema.columns.keys())
if extra_cols:
    logging.info(
        "Columns skipped by schema validation (strict=False): %s",
        sorted(extra_cols),
    )

# ------------------------------------------------------------------------------
# 3. VALIDATE WITH PANDERA (Polars backend)
# ------------------------------------------------------------------------------
logging.info("=== Validating with Pandera Schema ===")
try:
    df_pol = schema.validate(df_pol, lazy=True)
    logging.info("Data schema and ranges validated successfully with Pandera!")
except pa.errors.SchemaErrors as err:
    logging.warning("Pandera validation failed!")
    logging.warning(json.dumps(err.message, indent=2))
    logging.info("Continuing with unvalidated data.")


# ------------------------------------------------------------------------------
# 4. OUTLIER DETECTION (Z-Score) IN POLARS
# ------------------------------------------------------------------------------
# compute z-score columns for numeric columns
df_pol = df_pol.with_columns(
    [
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"{c}_z")
        for c in cfg.NUMERIC_COLUMNS
    ]
)
logging.info("=== Outlier Detection (Z-Score) ===")

Z = cfg.OUTLIER_ZSCORE_THRESHOLD
logging.info("Outlier Z-score threshold: (z > %.1f).", Z)


# outlier counts (1-row wide → long)
outlier_counts = df_pol.select(
    [(pl.col(f"{c}_z") > Z).sum().alias(c) for c in cfg.NUMERIC_COLUMNS]
).transpose(include_header=True, header_name="variable", column_names=["n_outliers"])

# non-null counts the same way
nonnull_counts = df_pol.select(
    [pl.col(c).is_not_null().sum().alias(c) for c in cfg.NUMERIC_COLUMNS]
).transpose(include_header=True, header_name="variable", column_names=["n_total"])

outlier_counts = outlier_counts.join(nonnull_counts, on="variable").with_columns(
    (pl.col("n_outliers") / pl.col("n_total") * 100).alias("pct")
)


if outlier_counts.is_empty():
    logging.info("No outliers detected in any columns (z > %.1f).", Z)
else:
    logging.warning("Columns with outliers (z > %.1f):", Z)
    for row in outlier_counts.iter_rows(named=True):
        logging.warning(
            "  - %s: %d rows (%.2f%%)", row["variable"], row["n_outliers"], row["pct"]
        )

# compute mean/std for each column
stats_list = [
    {"variable": c, "mean": float(df_pol[c].mean()), "std": float(df_pol[c].std())}
    for c in cfg.NUMERIC_COLUMNS
]
stats_df = pl.DataFrame(stats_list)

# merge stats into our summary
final_summary = outlier_counts.join(stats_df, on="variable")

# log every column, flagging nonzero outliers
for row in final_summary.iter_rows(named=True):
    msg = (
        f"{row['variable']}: mean={row['mean']:.3f}, std={row['std']:.3f}, "
        f"outliers={row['n_outliers']} ({row['pct']:.2f}%)"
    )
    if row["n_outliers"] > 0:
        logging.warning(msg)
    else:
        logging.info(msg)

# ------------------------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------------------------
logging.info("=== DATA QUALITY CHECK SUMMARY ===")
logging.info("Rows: %d, Columns: %d", df_pol.height, df_pol.width)
logging.info("Column dtypes: %s", df_pol.dtypes)

# --- Full numeric summary to CSV ------------------------------------------- #
summary_file = (
    Path(output_dir)
    / f"{Path(data_path).stem}_numeric_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
final_summary.write_csv(summary_file)
logging.info("Full numeric summary ➜ %s", summary_file)

# --- Numeric descriptive statistics (“describe”) to CSV --------------------- #
# Polars’ describe() gives you count, mean, std, min, max, quartiles, etc.
describe_df = df_pol.select(cfg.NUMERIC_COLUMNS).describe()
describe_file = (
    Path(output_dir)
    / f"{Path(data_path).stem}_numeric_describe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
describe_df.write_csv(describe_file)
logging.info("Numeric descriptive stats ➜ %s", describe_file)

logging.info("Data Quality Checks Complete.")
