"National statistics UPRNs lookup schema."

import polars as pl

schema = pl.Schema(
    [
        ("UPRN", pl.Int64),
        ("GRIDGB1E", pl.Int64),
        ("GRIDGB1N", pl.Int64),
        ("PCDS", pl.String),
        ("OA21CD", pl.String),
        ("cty25cd", pl.String),
        ("ced25cd", pl.String),
        ("lad25cd", pl.String),
        ("wd25cd", pl.String),
        ("hlth19cd", pl.String),
        ("ctry25cd", pl.String),
        ("rgn25cd", pl.String),
        ("pcon24cd", pl.String),
        ("eer20cd", pl.String),
        ("ttwa15cd", pl.String),
        ("itl25cd", pl.String),
        ("NPARK16CD", pl.String),
        ("lsoa21cd", pl.String),
        ("msoa21cd", pl.String),
        ("WZ11CD", pl.String),
        ("sicbl24cd", pl.String),
        ("bua24cd", pl.String),
        ("buasd11cd", pl.String),
        ("ruc21ind", pl.String),
        ("oac21ind", pl.String),
        ("lep21cd1", pl.String),
        ("lep21cd2", pl.String),
        ("pfa23cd", pl.String),
        ("imd19ind", pl.Int64),
    ]
)
