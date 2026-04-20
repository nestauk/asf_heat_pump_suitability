"""
Flow to calculate garden area (m2) where possible for properties in the domestic EPC register using Land Registry data and
Microsoft Building Footprints data.

To run:
python asf_heat_pump_suitability/pipeline/run_scripts/run_calculate_garden_size_flow.py --datastore=s3 run --epc [path/to/EPC/data] --year [YYYY] --quarter [Q] --nations ews

[Set --nations flag to "ew" or "s" for generating garden size estimates for either England and Wales or Scotland INSPIRE
files only.]

NB: this flow takes the preprocessed and deduplicated EPC dataset in parquet file format.
"""

from metaflow import FlowSpec, step, Parameter, batch


class CalculateGardenSizeFlow(FlowSpec):

    # Parameters
    epc = Parameter(
        name="epc",
        help="Path to processed and deduplicated EPC dataset in parquet file format",
        type=str,
        required=True,
    )

    year = Parameter(
        name="year",
        help="EPC data year. Format YYYY",
        type=int,
        required=True,
    )

    quarter = Parameter(
        name="quarter",
        help="EPC data quarter",
        type=int,
        required=True,
    )

    nations = Parameter(
        name="nations",
        help="Nations to get INSPIRE land registry file bounds for. Select from England and Wales (ew); Scotland (s); or all (ews).",
        type=str,
        required=True,
        default="ews",
    )

    sample = Parameter(
        name="sample",
        help="Run the script on a sample of 3 building footprint and land polygon file pairs",
        type=bool,
        required=False,
        default=False,
    )

    @step
    def start(self):
        """
        Load datasets and start flow.
        """
        import logging
        import polars as pl
        import geopandas as gpd
        from asf_heat_pump_suitability.utils import parallel_utils
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            building_footprint,
            lat_lon,
        )
        from asf_heat_pump_suitability.pipeline.transform import outdoor_space

        logging.info("Load EPC UPRNs")
        epc_df = pl.read_parquet(self.epc, columns=["UPRN"])

        logging.info("Adding lat/lon data to EPC")
        uprn_coords_df = lat_lon.transform_df_osopen_uprn_latlon()
        epc_df = epc_df.join(uprn_coords_df, how="left", on="UPRN")
        self.epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_df, usecols=["UPRN"])[
            ["UPRN", "geometry"]
        ]

        logging.info("Loading land registry file boundaries")
        land_file_bounds = gpd.read_file(
            f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/gardens/inspire_file_bounds_{self.nations.upper()}.geojson"
        )
        microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

        # Match land extent files with overlapping building footprint files
        file_matches = outdoor_space.match_series_files_land_building(
            land_files_gdf=land_file_bounds, building_files_gdf=microsoft_file_bounds
        )

        if self.sample:
            self.chunked_file_matches = parallel_utils.chunk_df(
                file_matches.sample(3, seed=1), size=1
            )
        else:
            self.chunked_file_matches = parallel_utils.chunk_df(file_matches, size=30)

        logging.info(
            f"Estimating garden size for properties across {len(file_matches)} pairs of land extent and building footprint files."
        )

        self.next(self.estimate_garden_size, foreach="chunked_file_matches")

    @batch(cpu=2, memory=16000)
    @step
    def estimate_garden_size(self):
        """
        Estimate garden size per property using land registry polygons and building footprints.
        """
        import os

        os.system(
            "pip install git+https://github.com/nestauk/asf_heat_pump_suitability.git"
        )

        import shapely
        import geopandas as gpd
        import polars as pl
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            building_footprint,
            land_extent,
        )
        from asf_heat_pump_suitability.pipeline.transform import outdoor_space

        prev = None
        self.epc_gardens = []

        for land_file, building_file in self.input.items():
            if land_file != prev:
                # Prepare land parcel data
                land_parcels_gdf = land_extent.transform_gdf_land_parcels(
                    f"s3://{land_file}"
                )

            # Prepare building footprints data
            try:
                building_footprints_gdf = (
                    building_footprint.transform_gdf_building_footprints(building_file)
                )
            except shapely.errors.GEOSException as e:
                print(
                    f"Error loading building footprint file {building_file}. Error message: {e}.\n"
                    f"Skipping this land extent & building footprint pairing."
                )
                continue
            else:
                building_footprints_gdf["microsoft_building_footprint_file"] = (
                    building_file
                )

                # Get intersection of building footprint polygons and land polygons
                intersection_gdf = outdoor_space.generate_gdf_building_intersections(
                    land_parcels_gdf=land_parcels_gdf,
                    building_gdf=building_footprints_gdf,
                )

                # Get garden size
                gardens_gdf = outdoor_space.generate_gdf_outdoor_space(
                    intersection_gdf, land_parcels_gdf
                )
                gardens_gdf = gardens_gdf.assign(
                    inspire_land_extent_file=land_file,
                    microsoft_building_footprint_file=building_file,
                )

                # Match EPC UPRNs with land parcels and gardens using UPRN coordinates
                # This will keep only EPC records for which garden size can be estimated
                epc_df = gpd.sjoin(
                    self.epc_gdf,
                    gardens_gdf,
                    how="inner",
                    predicate="intersects",
                ).drop(columns=["geometry", "index_right"])

                epc_df = pl.from_pandas(epc_df)
                self.epc_gardens.append(
                    epc_df.with_columns(
                        pl.col("NATIONALCADASTRALREFERENCE").cast(pl.String)
                    )
                )

        self.next(self.concatenate_garden_size_dfs)

    @step
    def concatenate_garden_size_dfs(self, inputs):
        """
        Concatenate estimated garden size data into one dataframe.
        """
        import itertools
        import polars as pl
        import logging

        self.epc_gardens_df = pl.concat(
            list(itertools.chain.from_iterable([input.epc_gardens for input in inputs]))
        )
        logging.info(
            f"Garden size calculated for {len(self.epc_gardens_df)} EPC properties in total."
        )
        self.next(self.save_outputs)

    # @batch(cpu=2, memory=16000)
    @step
    def save_outputs(self):
        """
        Save outputs to S3.
        """
        import polars as pl
        from asf_heat_pump_suitability.utils import save_utils
        from asf_heat_pump_suitability.pipeline.transform import outdoor_space

        save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/gardens/{self.year}_Q{self.quarter}_EPC_garden_size_estimates_{self.nations.upper()}.parquet"
        if self.sample:
            save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/gardens/{self.year}_Q{self.quarter}_EPC_garden_size_estimates_{self.nations.upper()}_SAMPLE.parquet"
        save_utils.save_to_s3(self.epc_gardens_df, save_as)

        self.epc_gardens_df = self.epc_gardens_df.with_columns(
            pl.col(pl.Float64).round(2)
        )
        self.epc_gardens_df = outdoor_space.deduplicate_df_outdoor_space(
            self.epc_gardens_df
        )

        save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/gardens/{self.year}_Q{self.quarter}_EPC_garden_size_estimates_{self.nations.upper()}_deduplicated.parquet"
        if self.sample:
            save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/gardens/{self.year}_Q{self.quarter}_EPC_garden_size_estimates_{self.nations.upper()}_deduplicated_SAMPLE.parquet"
        save_utils.save_to_s3(self.epc_gardens_df, save_as)

        self.next(self.end)

    @step
    def end(self):
        """
        Finish flow.
        """
        import logging

        logging.info("Calculate garden size flow complete!")


if __name__ == "__main__":
    CalculateGardenSizeFlow()
