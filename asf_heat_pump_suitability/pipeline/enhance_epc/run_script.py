import logging
import polars as pl
import s3fs
import pathlib
from typing import Optional
from argparse import ArgumentParser
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.enhance_epc import enhance_epc
from asf_heat_pump_suitability.pipeline.reweight_epc import reweight_epc


def run():
    """
    Create ArgumentParser and passes arguments to `main()` and runs `main()`.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--epc_path", help="S3 URI to EPC dataset", type=str, required=True
    )

    parser.add_argument(
        "--save_output",
        help="S3 path to save enhanced EPC dataset to",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args()

    main(**vars(args))


def main(epc_path: str, save_output: Optional[str]) -> pl.DataFrame:
    """
    Enhance EPC dataset with additional features: LSOA; MSOA.

    Args
        epc_path (str): S3 URI to EPC dataset
        save_output (str): S3 path to save enhanced EPC dataset to. Optional.

    Returns
        pl.DataFrame: enhanced EPC dataset with additional features
    """
    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    if pathlib.Path(epc_path).suffixes == ".csv":
        fs = s3fs.S3FileSystem()
        with fs.open(epc_path, mode="rb") as f:
            epc_df = pl.read_csv(f, columns=config["usecols"]["epc"])
    else:
        epc_df = pl.read_parquet(epc_path, columns=config["usecols"]["epc"])

    # Join ONSPD LSOA col
    enhanced_epc_df = enhance_epc.join_df_additional_features(epc_df)
    # Prepare EPC df for reweighting
    enhanced_epc_df = reweight_epc.add_cols_weighting_features(enhanced_epc_df)

    # Save to S3
    if save_output:
        fs = s3fs.S3FileSystem()
        with fs.open(save_output, mode="wb") as f:
            enhanced_epc_df.write_parquet(f)

    return enhanced_epc_df


if __name__ == "__main__":
    run()
