import logging


def transform_gdf_drop_close_duplicates(
    gdf,
):  # TODO: this is dropping duplicate geometries within one file. Is there a reason we shouldnt do this?
    """
    Drop polygons with the same representative point. This should drop duplicates and almost duplicates
    """
    gdf["rep_point"] = gdf.representative_point()
    if not gdf["rep_point"].nunique() == len(gdf):
        dups_count = gdf.duplicated(subset="rep_point").sum()
        gdf = gdf.sort_values(by="geometry").drop_duplicates(
            subset="rep_point", keep="first"
        )
        logging.info(
            f"Polygons containing same representative point found. "
            f"Dropping {dups_count} polygons."
        )

    return gdf
