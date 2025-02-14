import polars as pl
import boto3
import os
import plotnine as pn
import numpy as np
import requests

session = boto3.Session(profile_name="nesta")
credentials = session.get_credentials()
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
os.environ["AWS_SESSION_TOKEN"] = credentials.token

df_desnz = pl.read_parquet(
    "s3://asf-heat-pump-suitability/evaluation/desnz_hn_zone_scores/hp_suitability_scores_with_desnz/*.parquet"
)
df_lsoa = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_lsoa.parquet"
)

# get ASHP scores alongside DESNZ scores
df_desnz_full = df_desnz.join(
    df_lsoa.select(["lsoa", "ASHP_N_avg_score_weighted"]),
    how="left",
    left_on="LSOA21CD",
    right_on="lsoa",
)

# remove DESNZ LSOAs from other data frame
df_lsoa_dropped = df_lsoa.join(
    df_desnz, how="anti", left_on="lsoa", right_on="LSOA21CD"
)

# create bins for mapping
df_desnz_plot = (
    df_desnz_full.with_columns(
        [
            pl.col("ASHP_N_avg_score_weighted").cut(
                breaks=np.arange(0, 1.05, 0.05),
                labels=[str(round(x, 3)) for x in np.arange(0.025, 1.1, 0.05)],
            ),
            pl.col("HN_N_avg_score_weighted").cut(
                breaks=np.arange(0, 1.05, 0.05),
                labels=[str(round(x, 3)) for x in np.arange(0.025, 1.1, 0.05)],
            ),
        ]
    )
    .group_by(["ASHP_N_avg_score_weighted", "HN_N_avg_score_weighted"])
    .agg(
        pl.col("DESNZ_pilot_fraction").mean().alias("DESNZ_mean"),
        (pl.col("DESNZ_pilot_fraction") > 0).mean().alias("DESNZ_present"),
    )
    .cast(
        {"ASHP_N_avg_score_weighted": pl.Float32, "HN_N_avg_score_weighted": pl.Float32}
    )
)

# map the heatmap first

p_mean = (
    pn.ggplot(
        mapping=pn.aes(x="HN_N_avg_score_weighted", y="ASHP_N_avg_score_weighted")
    )
    + pn.geom_tile(
        mapping=pn.aes(fill="DESNZ_mean", width=0.05, height=0.05),
        data=df_desnz_plot,
    )
    + pn.scale_fill_gradient(low="#e4f6f8", high="darkblue")
    + pn.scale_x_continuous(breaks=np.arange(0, 1.1, 0.1))
    + pn.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1))
    + pn.theme_minimal()
    + pn.theme(plot_background=pn.element_rect(fill="#f8f5f4"))
    + pn.labs(
        x="HN suitability",
        y="ASHP suitability",
        fill="Mean DESNZ fraction",
        title="Comparing LSOAs in DESNZ HN zones",
    )
)

p_mean

# now look with the points plotted on top
(p_mean + pn.geom_point(data=df_lsoa_dropped, alpha=1, fill="black", color="white"))

# watch out for artifacts in our scoring system, need to be careful plotting these

(p_mean + pn.geom_point(data=df_lsoa_dropped, alpha=0.1, fill="black", color="white"))


# try the same plot but looking at DESNZ present or not (binary), rather than average

p_present = (
    pn.ggplot(
        mapping=pn.aes(x="HN_N_avg_score_weighted", y="ASHP_N_avg_score_weighted")
    )
    + pn.geom_tile(
        mapping=pn.aes(fill="DESNZ_present", width=0.05, height=0.05),
        data=df_desnz_plot,
    )
    + pn.scale_fill_gradient(
        low="#e4f6f8",
        high="darkblue",
        labels=lambda x: [f"{val*100:.0f}%" for val in x],
    )
    + pn.scale_x_continuous(breaks=np.arange(0, 1.1, 0.1))
    + pn.scale_y_continuous(breaks=np.arange(0, 1.1, 0.1))
    + pn.theme_minimal()
    + pn.theme(plot_background=pn.element_rect(fill="#f8f5f4"))
    + pn.labs(
        x="HN suitability",
        y="ASHP suitability",
        fill="% DESNZ present",
        title="Comparing LSOAs in DESNZ HN zones",
    )
)

p_present

(
    p_present
    + pn.geom_point(data=df_lsoa_dropped, alpha=0.1, fill="black", color="white")
)

# get data necessary to map LSOA codes to LA in E/W and S

df_lsoa_map = pl.read_csv(
    "https://hub.arcgis.com/api/v3/datasets/686527814d73403e8f0a59c7a28b0c34_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1"
)
df_dz_map = pl.read_csv(
    "https://scottish-government-files.s3.eu-west-1.amazonaws.com/50d30936-6de2-4ae0-a131-c2105aa74647/DataZone2011lookup_2024-12-16.csv",
    encoding="latin-1",
)

df_map = pl.concat(
    [
        df_lsoa_map[["LSOA21CD", "LAD24CD", "LAD24NM"]],
        df_dz_map.select(
            pl.col("DZ2011_Code").alias("LSOA21CD"),
            pl.col("LA_Code").alias("LAD24CD"),
            pl.col("LA_Name").alias("LAD24NM"),
        ),
    ]
)

# join up to our dataset that was not mapped by DESNZ
# and calculate the average scores for each LA

df_la = (
    df_lsoa_dropped.join(df_map, left_on="lsoa", right_on="LSOA21CD", how="left")
    .group_by(["LAD24CD", "LAD24NM"])
    .agg(
        pl.col("ASHP_N_avg_score_weighted").median(),
        pl.col("HN_N_avg_score_weighted").median(),
    )
)

# plot LAs on top of the map

(p_present + pn.geom_point(data=df_la, alpha=1, color="white", fill="black"))
