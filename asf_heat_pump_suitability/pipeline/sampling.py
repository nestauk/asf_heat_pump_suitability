import polars as pl
from typing import Dict, List, Tuple
import s3fs
import logging


def filter_df_area_sample_epc(
    df: pl.DataFrame,
    area: str,
    country_codes: list = ["E", "S", "W"],
    min_count: int = 10,
) -> pl.DataFrame:
    """
    Filter EPC dataset to sampled areas with min and max count of UPRNs.

    Args
        df (pl.DataFrame): EPC dataset
        area (str): area-type to sample dataset by
        min_count (int): minimum count of UPRNs in areas selected. Default `10`
        country_codes (list): country codes to sample from. Default `["E", "S", "W"]`

    Returns
        pl.DataFrame: subset of EPC dataset with sampled areas only
    """
    min, max = get_dicts_minmax_sample_per_nation(df, area, country_codes, min_count)
    sample_areas = [d.get(area) for d in min]
    sample_areas.extend([d.get(area) for d in max])

    return df.filter(pl.col(area).is_in(sample_areas))


def get_dicts_minmax_sample_per_nation(
    df: pl.DataFrame, area: str, country_codes: list, min_count: int
) -> Tuple[List[Dict]]:
    """
    Get sample areas from EPC dataset. Get areas in each nation with min and max count of UPRNs.

    Args
        df (pl.DataFrame): EPC dataset
        area (str): name of area-type column to sample dataset by
        min_count (int): minimum count of UPRNs in areas selected
        country_codes (list): country codes to sample from

    Returns
        tuple[list[dict]]: dicts containing sample information for two sample areas per nation: min and max count of UPRNs.
    """
    _count = df.group_by([area, "country_code"]).agg(pl.col("UPRN").count())
    _count = _count.filter(pl.col("UPRN") >= min_count)

    # Get min and max UPRN counts
    _min = _count.filter(UPRN=pl.col("UPRN").min().over("country_code"))
    _max = _count.filter(UPRN=pl.col("UPRN").max().over("country_code"))

    return _get_dicts_sample_per_nation(
        _min, country_codes
    ), _get_dicts_sample_per_nation(_max, country_codes)


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

    from asf_heat_pump_suitability.pipeline import run_script

    epc_path = (
        "s3://asf-daps/lakehouse/processed/epc/deduplicated/processed_dedupl-0.parquet"
    )
    epc = run_script.main(epc_path=epc_path, save_output=None)

    # Sample by LSOA & MSOA and save
    for oa in ["lsoa", "msoa"]:
        sample = filter_df_area_sample_epc(epc, area=oa)
        fs = s3fs.S3FileSystem()
        save_path = f"s3://asf-heat-pump-suitability/outputs/epc_sample_{oa}.parquet"
        logging.info(f"Saving {oa} sample to: {save_path}")
        with fs.open(save_path, mode="wb") as f:
            sample.write_parquet(f)
