"""
Add new features to EPC dataset:
- mean average garden size per MSOA
- lat/lon per UPRN
- property density per LSOA
- off gas properties by postcode
- listed building status per UPRN
- protected area flag per UPRN which includes properties in England and Wales building conservation areas & Scotland World Heritage Sites
- grid capacity per LSOA (% of homes which could install a HP with current grid capacity)
- presence of anchor properties per LSOA

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_add_features_flow.py --datastore=s3 run --epc_path [path/to/EPC] --year [YYYY] --quarter [Q]

NB: this pipeline takes the preprocessed and deduplicated EPC dataset in parquet file format.
"""

from metaflow import FlowSpec, step, Parameter


class AddFeaturesFlow(FlowSpec):
    epc_path = Parameter(
        name="epc_path",
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

    @step
    def start(self):
        """
        Load EPC data and start flow.
        """
        import logging
        import polars as pl

        # Import processed EPC
        logging.info(f"Loading EPC file from path: {self.epc_path}")
        self.epc_df = pl.read_parquet(
            self.epc_path,
            columns=[
                "UPRN",
                "COUNTRY",
                "POSTCODE",
                "PROPERTY_TYPE",
                "BUILT_FORM",
                "CURRENT_ENERGY_RATING",
            ],
        ).with_columns(
            pl.when(pl.col("UPRN").str.contains(r"[a-zA-Z]"))
            .then(False)
            .otherwise(True)
            .alias("valid_UPRN")
        )

        self.next(self.clean_property_type)

    @step
    def clean_property_type(self):
        """
        Clean property type column to create four categories of detached, semi-detached, terraced, and flats.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_sample

        logging.info("Cleaning property type data in EPC")
        # Below we process the EPC `PROPERTY_TYPE` column and rename, then drop the original column
        self.epc_df = prepare_sample.add_col_property_type(self.epc_df).drop(
            ["PROPERTY_TYPE", "BUILT_FORM"]
        )

        self.next(self.add_output_areas)

    @step
    def add_output_areas(self):
        """
        Add LSOA, MSOA, LAD, and rural-urban indicators to EPC.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import output_areas

        logging.info("Adding LSOA, MSOA, LAD, and rural-urban indicators to EPC")
        onspd_df = output_areas.load_transform_df_area_info()

        self.epc_df = output_areas.standardise_col_postcode(
            self.epc_df, pcd_col="POSTCODE"
        )
        self.epc_df = self.epc_df.join(onspd_df, how="left", on="POSTCODE")

        self.next(self.add_lat_lon_data)

    @step
    def add_lat_lon_data(self):
        """
        Add lat/ lon data to EPC per property.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            lat_lon,
            output_areas,
        )

        logging.info("Adding lat/lon data to EPC")
        uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
        self.epc_df = self.epc_df.join(uprn_latlon_df, how="left", on="UPRN")
        self.epc_gdf = lat_lon.generate_gdf_uprn_coords(
            self.epc_df, usecols=["UPRN", "COUNTRY", "lad_code"]
        )

        # TODO this is far too slow and is only used to correct some EPC records with incorrect postcodes which get joined to the wrong LSOA/MSOA
        # TODO can we filter the dataset somehow and only do the join with incorrect postcodes?
        # Replace `lad_code` from postcode with `lad_code` from geospatial join and postcode
        logging.info("Adding LAD code with geospatial join")
        uprn_lad_df = output_areas.sjoin_df_uprn_lad_code(self.epc_gdf)
        self.epc_df = self.epc_df.drop("lad_code").join(
            uprn_lad_df, how="left", on="UPRN"
        )

        self.next(self.add_protected_area_flag)

    @step
    def add_protected_area_flag(self):
        """
        Add protected area flag.
        """
        import logging
        import polars as pl
        from asf_heat_pump_suitability.pipeline.prepare_features import protected_areas

        logging.info("Adding protected area flag")
        uprns_in_protected_area_df = (
            protected_areas.load_transform_df_uprn_in_protected_area(self.epc_gdf)
        )
        self.epc_df = self.epc_df.join(
            uprns_in_protected_area_df, how="left", on="UPRN"
        )

        logging.info(
            "Adding local authority building conservation area data availability flag for England and Wales"
        )
        lad_cons_areas_df = (
            protected_areas.generate_df_conservation_area_data_availability(
                ladcd_col="LAD23CD"
            )
        )
        self.epc_df = self.epc_df.join(
            lad_cons_areas_df, how="left", left_on="lad_code", right_on="LAD23CD"
        )

        self.epc_df = self.epc_df.with_columns(
            pl.when(
                (pl.col("lad_conservation_area_data_available_ew"))
                & pl.col("COUNTRY").is_in(["England", "Wales"])
                & pl.col("valid_UPRN")
            )
            .then(pl.col("in_protected_area").fill_null(False))
            .otherwise(pl.col("in_protected_area"))
            .alias("in_protected_area")
        )

        self.next(self.add_average_garden_size_per_msoa)

    @step
    def add_average_garden_size_per_msoa(self):
        """
        Add average garden size per MSOA to EPC.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            epc,
            garden_space_avg,
        )

        logging.info("Adding average garden size per MSOA to EPC")
        garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()
        self.epc_df = epc.add_col_msoa_avg_outdoor_space_property_type(self.epc_df)
        self.epc_df = self.epc_df.join(
            garden_space_avg_msoa_df,
            how="left",
            left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
            right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
        )

        self.next(self.add_property_density)

    @step
    def add_property_density(self):
        """
        Add property density per LSOA.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import property_density

        logging.info("Adding property density to EPC")
        lsoa_density_df = property_density.generate_df_property_density()
        self.epc_df = self.epc_df.join(lsoa_density_df, how="left", on="lsoa")

        self.next(self.add_off_gas_flag)

    @step
    def add_off_gas_flag(self):
        """
        Add off gas grid column to EPC per postcode.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import off_gas

        logging.info("Adding off gas grid column to EPC")
        off_gas_postcodes = off_gas.process_off_gas_data()
        self.epc_df = off_gas.add_off_gas_feature(self.epc_df, off_gas_postcodes)

        self.next(self.add_listed_building_status)

    @step
    def add_listed_building_status(self):
        """
        Add flag indicating whether property is within a listed building.
        """
        import logging
        import polars as pl
        from asf_heat_pump_suitability.pipeline.prepare_features import listed_buildings

        logging.info("Adding listed buildings to EPC")
        listed_buildings_df = listed_buildings.generate_df_epc_listed_buildings(
            epc_df=self.epc_df
        )
        self.epc_df = self.epc_df.join(
            listed_buildings_df, how="left", on="UPRN"
        ).with_columns(
            # Only fill nulls of valid UPRNs because invalid UPRNs have no coordinate data so we don't know if they're in a listed building or not
            pl.when(pl.col("valid_UPRN"))
            .then(pl.col("listed_building").fill_null(False))
            .otherwise(pl.col("listed_building"))
            .alias("listed_building")
        )

        self.next(self.add_grid_capacity)

    @step
    def add_grid_capacity(self):
        """
        Add LSOA grid capacity and maximum percentage of households per LSOA that the grid could support having a heat
        pump.
        """
        import logging
        from asf_heat_pump_suitability.pipeline.prepare_features import grid_capacity

        logging.info("Adding grid capacity column to EPC")
        grid_capacities = grid_capacity.calculate_grid_capacity().select(
            ["lsoa", "heatpump_installation_percentage"]
        )
        self.epc_df = self.epc_df.join(grid_capacities, how="left", on="lsoa")

        self.next(self.add_anchor_property_flag)

    @step
    def add_anchor_property_flag(self):
        """
        Add flag to indicate whether LSOA has at least one anchor property.
        """
        import logging
        import polars as pl
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            anchor_properties,
        )

        logging.info("Adding anchor properties column to EPC")
        anchor_properties_df = anchor_properties.identify_anchor_properties_df().select(
            ["lsoa", "has_anchor_property"]
        )
        self.epc_df = self.epc_df.join(
            anchor_properties_df, how="left", on="lsoa"
        ).with_columns(pl.col("has_anchor_property").fill_null(False))

        self.next(self.save_output)

    @step
    def save_output(self):
        """
        Save final dataframe to S3.
        """
        from asf_heat_pump_suitability.utils import save_utils

        save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/features/{self.year}_Q{self.quarter}_EPC_features.parquet"
        save_utils.save_to_s3(self.epc_df, save_as)

        self.next(self.end)

    @step
    def end(self):
        """
        Finish flow.
        """
        import logging

        logging.info("Add features flow complete!")


if __name__ == "__main__":
    AddFeaturesFlow()
