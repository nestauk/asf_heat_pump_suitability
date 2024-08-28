import polars as pl
from typing import Dict, List
import s3fs
import logging
import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse arguments.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epc_weights_path",
        help="Path to EPC with weights parquet file",
        type=str,
        required=True,
    )
    return parser.parse_args()


def filter_df_area_sample_epc(
    df: pl.DataFrame,
    area: str,
    country_codes: list = ["E", "S", "W"],
    min_count: int = 10,
    quantiles: List[int] = [0, 0.25, 0.5, 0.75, 1],
) -> pl.DataFrame:
    """
    Filter EPC dataset to sampled areas with count of UPRNs cut at specified quantiles.

    Args
        df (pl.DataFrame): EPC dataset
        area (str): area-type to sample dataset by
        min_count (int): minimum count of UPRNs in areas selected. Default `10`
        country_codes (list): country codes to sample from. Default `["E", "S", "W"]`
        quantiles (List[int]): quantiles to sample from. Value 0 <= x <= 1.

    Returns
        pl.DataFrame: subset of EPC dataset with sampled areas only
    """
    _dicts = get_dicts_quantile_sample_per_nation(
        df=df,
        area=area,
        country_codes=country_codes,
        min_count=min_count,
        quantiles=quantiles,
    )
    sample_areas = [d.get(area) for d in _dicts]

    return df.filter(pl.col(area).is_in(sample_areas))


def get_dicts_quantile_sample_per_nation(
    df: pl.DataFrame,
    area: str,
    country_codes: list,
    min_count: int,
    quantiles: List[int],
) -> List[Dict]:
    """
    Get sample areas from EPC dataset. Get areas in each nation with count of UPRNs cut at specified quantiles.

    Args
        df (pl.DataFrame): EPC dataset
        area (str): name of area-type column to sample dataset by
        country_codes (list): country codes to sample from
        min_count (int): minimum count of UPRNs in areas selected
        quantiles (List[int]): quantiles to sample from. Value 0 <= x <= 1.

    Returns
        list[dict]: dicts containing sample information for each sample area per nation at specified quantiles
    """
    _count = df.group_by([area, "country_code"]).agg(pl.col("UPRN").count())
    _count = _count.filter(pl.col("UPRN") >= min_count).sort(area)

    sample_areas = []
    for q in quantiles:
        _df = _count.filter(
            UPRN=pl.col("UPRN")
            .quantile(quantile=q, interpolation="nearest")
            .over("country_code")
        )
        sample_areas.extend(
            _get_dicts_sample_per_nation(_df, country_codes=country_codes)
        )

    return sample_areas


def _get_dicts_sample_per_nation(df: pl.DataFrame, country_codes: list) -> List[Dict]:
    """
    Get one sample per nation.

    Args
        df (pl.DataFrame): dataset to sample from
        country_codes (list): list of country codes to sample from

    Returns
        list[dict]: list of dicts containing sample information
    """
    return [
        df.filter(pl.col("country_code") == cc).sample(1, seed=8).to_dicts()[0]
        for cc in country_codes
    ]


if __name__ == "__main__":
    args = parse_arguments()
    epc_path = args.epc_weights_path

    # Load EPC data with weights
    epc = pl.read_parquet(epc_path)

    # Sample by LSOA & MSOA and save
    for oa in ["lsoa", "msoa"]:
        sample = filter_df_area_sample_epc(epc, area=oa)
        fs = s3fs.S3FileSystem()
        save_path = f"s3://asf-heat-pump-suitability/outputs/epc_sample_{oa}.parquet"
        logging.info(f"Saving {oa} sample to: {save_path}")
        with fs.open(save_path, mode="wb") as f:
            sample.write_parquet(f)
