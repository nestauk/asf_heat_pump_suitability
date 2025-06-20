"""
Flow to weight properties with Iterative Proportional Fitting per LSOA / Data Zone according to the
following features:
- property type (detached, semi-detached, terraced, flats, other);
- tenure (owner-occupied, social rental, private rental)
- build year (pre- and post-1930 split, and unknown); [applies to England and Wales only*]

*Data Zones in Scotland are the closest equivalent to LSOAs in England and Wales. They are reweighted on two features
only (property type and tenure) because there is no target build year data aggregated to Data Zone-level available for
Scotland.

To run:
python asf_heat_pump_suitability/pipeline/run_scripts/run_compute_epc_weights.py --datastore=s3 run --epc [path/to/EPC] --year [YYYY] --quarter [Q]

NB: this pipeline takes the preprocessed and deduplicated EPC dataset in parquet file format.
"""

from metaflow import FlowSpec, step, batch, Parameter


class ComputeEpcWeightsFlow(FlowSpec):
    """
    Flow to weight properties with Iterative Proportional Fitting per LSOA / Data Zone.
    """

    epc_path = Parameter(
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
        help="EPC data quarter, 1-4",
        type=int,
        required=True,
    )

    sample = Parameter(
        name="sample",
        help="Set to True to sample 1000 rows from the EPC dataset to run the flow on. Defaults to False which runs the flow on the full EPC dataset.",
        type=bool,
        default=False,
        required=False,
    )

    @step
    def start(self):
        """
        Load EPC data and start flow.
        """
        import polars as pl

        # Set reweighting features for each nation
        self.country_features = [
            ("Scotland", ["property_type", "tenure"]),
            ("England", ["property_type", "tenure", "build_year"]),
            ("Wales", ["property_type", "tenure", "build_year"]),
        ]

        # Import processed & deduplicated EPC
        print(f"Loading EPC file from path: {self.epc_path}")
        self.epc_df = pl.read_parquet(
            self.epc_path,
            columns=[
                "UPRN",
                "POSTCODE",
                "COUNTRY",
                "TENURE",
                "PROPERTY_TYPE",
                "BUILT_FORM",
                "CONSTRUCTION_AGE_BAND",
            ],
        )

        if self.sample:
            print("Running ComputeEpcWeightsFlow on a sample of EPC data (N=1000)")
            self.epc_df = self.epc_df.sample(n=1000, seed=2)
            self.batch_memory = 1000
        else:
            self.batch_memory = 16000

        print(f"Setting memory for batch steps to {self.batch_memory} MB")

        self.next(self.join_lsoa_code)

    @step
    def join_lsoa_code(self):
        """
        Join LSOA code to each record in EPC.
        """
        from asf_heat_pump_suitability.pipeline.prepare_features import output_areas

        # Join ONS Postcode Directory LSOA col
        self.epc_df = output_areas.standardise_col_postcode(
            self.epc_df, pcd_col="POSTCODE"
        )
        lsoa_df = output_areas.load_transform_df_lsoas()
        self.epc_df = self.epc_df.join(lsoa_df, how="left", on="POSTCODE")
        self.next(self.prepare_for_reweighting)

    @step
    def prepare_for_reweighting(self):
        """
        Standardise EPC features used in weighting and drop EPC rows missing data required for reweighting (those missing
        LSOA information or reweighting features).
        """
        from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_sample

        self.epc_df = self.epc_df.drop_nulls(subset=["lsoa"])
        self.epc_df = prepare_sample.add_cols_weighting_features(self.epc_df)
        self.next(
            self.prepare_for_country_specific_reweighting, foreach="country_features"
        )

    # @batch(cpu=2, memory=16000)
    @batch(cpu=2, memory=1000)
    @step
    def prepare_for_country_specific_reweighting(self):
        """
        For each country, conduct country-specific preprocessing on the EPC data to prepare for reweighting.
        """
        # TODO update to dev branch before merge
        # Install repo on batch machine to access modules
        import os

        os.system(
            "pip install git+https://github.com/nestauk/asf_heat_pump_suitability.git@154_parallelise_reweighting"
        )
        import polars as pl
        from asf_heat_pump_suitability.utils import parallel_utils
        from asf_heat_pump_suitability.pipeline.reweight_epc import (
            prepare_sample,
            prepare_target,
        )

        country, self.features = self.input
        print(
            f"Running reweighting for {country}. Reweighting using the following features: {self.features}"
        )
        epc_cleaned_df = self.epc_df.filter(pl.col("COUNTRY") == country)
        assert len(epc_cleaned_df) > 0, f"No EPC records found for {country}."

        self.epc_cleaned_df = prepare_sample.drop_nulls_feature_cols(
            df=epc_cleaned_df, features=self.features
        )

        if self.sample:
            self.chunks = parallel_utils.chunk_df_by_group(
                self.epc_cleaned_df, group_col="lsoa", n=100
            )
        else:
            self.chunks = parallel_utils.chunk_df_by_group(
                self.epc_cleaned_df, group_col="lsoa", n=1000
            )

        # Generate target marginals for all features and LSOAs
        self.target_marginals = prepare_target.get_dict_target_marginals(
            features=self.features
        )

        self.next(self.reweight_properties_per_lsoa, foreach="chunks")

    # @batch(cpu=2, memory=16000)
    @batch(cpu=2, memory=1000)
    @step
    def reweight_properties_per_lsoa(self):
        """
        For each chunk of EPC data per country, use Iterative Proportional Fitting (IPF) to calculate weights for all
        properties per LSOA / DZ. Properties are weighted so that the total proportions of each target feature match as
        closely as possible to the target marginals of each target feature in the census data for the LSOA / DZ.

        This step also saves information about how long each LSOA / DZ takes to reweight and how many EPC rows are
        not weighted due to preprocessing.
        """
        # TODO update to dev branch before merge
        # Install repo on batch machine to access modules
        import os

        os.system(
            "pip install git+https://github.com/nestauk/asf_heat_pump_suitability.git@154_parallelise_reweighting"
        )
        from tqdm import tqdm
        import time
        from asf_heat_pump_suitability.pipeline.reweight_epc import (
            prepare_target,
            reweight_epc,
        )

        # Prepare results dicts
        self.lsoa = []
        self.uprn = []
        self.lsoas = []
        self.weight = []
        self.proportional_weight = []
        self.time = []
        self.lost_rows = []

        for lsoa in tqdm(self.input["lsoa"].unique()):
            self.lsoa.append(lsoa)

            try:
                start = time.time()
                sample, lost_rows = reweight_epc.generate_balance_sample(
                    df=self.input,
                    features=self.features,
                    lsoa=lsoa,
                    target_marginals=self.target_marginals,
                )

                if sample:
                    target = prepare_target.generate_balance_target_population(
                        target_marginals=self.target_marginals, lsoa=lsoa
                    )
                    weighted_sample = reweight_epc.generate_weighted_sample(
                        balance_sample=sample, balance_target=target
                    )
                    _weights = reweight_epc.get_dict_sample_weights(
                        weighted_sample=weighted_sample
                    )

                    # Add output weights for LSOA to dict
                    self.uprn.extend(_weights["UPRN"])
                    # Adding LSOA required for dummy rows
                    self.lsoas.extend([lsoa for i in range(len(_weights["UPRN"]))])
                    self.weight.extend(_weights["weight"])
                    self.proportional_weight.extend(_weights["proportional_weight"])

                else:
                    print(
                        f"No records remaining to reweight after preprocessing for LSOA: {lsoa}. Skipping."
                    )

                # LSOA stats
                end = time.time()
                self.time.append(end - start)
                self.lost_rows.append(lost_rows)

            except KeyError:
                print(f"No target data found for LSOA: {lsoa}. Skipping.")
                self.time.append(None)
                self.lost_rows.append(None)
                continue

        self.next(self.join_weights)

    @step
    def join_weights(self, inputs):
        """
        Join chunks of weighted EPC data and weighting stats together per country.
        """
        import polars as pl
        import itertools

        # This line exists to persist the epc_df variable to the next step in the flow
        self.epc_df = inputs[0].epc_df

        # Get df of UPRNs, reweighting features, and weights for all nations
        self.weights_df = pl.DataFrame(
            {
                "UPRN": list(
                    itertools.chain.from_iterable([input.uprn for input in inputs])
                ),
                "lsoa": list(
                    itertools.chain.from_iterable([input.lsoas for input in inputs])
                ),
                "weight": list(
                    itertools.chain.from_iterable([input.weight for input in inputs])
                ),
                "proportional_weight": list(
                    itertools.chain.from_iterable(
                        [input.proportional_weight for input in inputs]
                    )
                ),
            },
            schema={
                "UPRN": pl.String,
                "lsoa": pl.String,
                "weight": pl.Float64,
                "proportional_weight": pl.Float64,
            },
        )

        # Get df of stats for all nations
        self.lsoa_stats_df = pl.DataFrame(
            {
                "lsoa": list(
                    itertools.chain.from_iterable([input.lsoa for input in inputs])
                ),
                "time": list(
                    itertools.chain.from_iterable([input.time for input in inputs])
                ),
                "lost_rows": list(
                    itertools.chain.from_iterable([input.lost_rows for input in inputs])
                ),
            },
            schema={"lsoa": pl.String, "time": pl.Float64, "lost_rows": pl.Float64},
        )

        self.next(self.join_countries)

    @step
    def join_countries(self, inputs):
        """
        Concatenate weighted EPC data from each country together into single dataframe.
        """
        import polars as pl

        self.epc_df = inputs[0].epc_df
        self.weights_df = pl.concat([input.weights_df for input in inputs])
        self.lsoa_stats_df = pl.concat([input.lsoa_stats_df for input in inputs])
        self.next(self.join_weights_to_epc)

    @step
    def join_weights_to_epc(self):
        """
        Merge weighted EPC with reweighting features.
        """
        # Left join ensures we retain dummy rows which we need to retain for reweighting evaluation
        self.epc_df = self.epc_df.select(
            ["UPRN", "property_type", "tenure", "build_year"]
        )
        self.weights_df = self.weights_df.join(self.epc_df, how="left", on="UPRN")
        self.next(self.save_results)

    @step
    def save_results(self):
        """
        Save reweighted EPC data and reweighting stats to S3.
        """
        from asf_heat_pump_suitability.utils import save_utils

        save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/weights/{self.year}_Q{self.quarter}_EPC_weights"
        if self.sample:
            save_as = save_as + "_SAMPLE"
        save_utils.save_to_s3(self.weights_df, f"{save_as}.parquet")
        save_utils.save_to_s3(self.lsoa_stats_df, f"{save_as}_stats.parquet")
        self.next(self.end)

    @step
    def end(self):
        """
        End flow.
        """
        print("Compute EPC weights flow complete!")


if __name__ == "__main__":
    ComputeEpcWeightsFlow()
