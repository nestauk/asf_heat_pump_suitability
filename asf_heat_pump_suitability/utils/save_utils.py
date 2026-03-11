import logging
import pickle
from pathlib import Path

import polars as pl
import s3fs
from sklearn.base import BaseEstimator

from asf_heat_pump_suitability.config.settings import Settings

logger = logging.getLogger(__name__)


def save_df(df: pl.DataFrame, filename: str, settings: Settings) -> None:
    """Save a DataFrame to the configured output directory (local or S3).

    The output path is resolved from ``filename`` using ``settings``. Supports
    .parquet and .csv. Local paths have their parent directory created
    automatically if it does not already exist.

    Args:
        df (pl.DataFrame): DataFrame to save.
        filename (str): Output filename (e.g. ``domestic_uprns.parquet``).
        settings (Settings): Pipeline settings used to resolve the output path.
    """
    path = settings.resolve_output_path(filename)
    print(f"Saving to: {path}")
    file_type = path.split(".")[-1]
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        if file_type == "parquet":
            with fs.open(path=path, mode="wb") as f:
                df.write_parquet(f)
        elif file_type == "csv":
            with fs.open(path=path, mode="wb") as f:
                df.write_csv(f)
        else:
            raise ValueError(f"Unsupported file type for S3 save: .{file_type}")
    else:
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        if file_type == "parquet":
            df.write_parquet(path)
        elif file_type == "csv":
            df.write_csv(path)
        else:
            raise ValueError(f"Unsupported file type for local save: .{file_type}")


def save_model_to_pkl(model: BaseEstimator, path: str) -> None:
    """Save a fitted scikit-learn estimator as a pickle file to S3 or local filesystem.

    Local paths have their parent directory created automatically if it does not
    already exist.

    Args:
        model (BaseEstimator): Trained estimator to persist.
        path (str): Destination path — either a local filesystem path or an
            ``s3://`` URI.
    """
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        with fs.open(path, "wb") as f:
            pickle.dump(model, f)
    else:
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    print(f"Saved model to {path}")
