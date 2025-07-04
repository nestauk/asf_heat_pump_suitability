"""
Calculate suitability scores of different low-carbon heating technologies in Nesta and 'conventional' views for individual
properties and LSOAs.

To run:
python asf_heat_pump_suitability/pipeline/flows/run_calculate_suitability_flow.py --datastore=s3 run --weights [path/to/weighted/EPC] --features [path/to/EPC/with/features] --gardens [path/to/garden/size/estimates] --year [YYYY] --quarter [Q] --max-workers 18

Set --sample to True to run on a sample of 1000 EPC records

NB: this pipeline takes the outputs from the following scripts as inputs:
- asf_heat_pump_suitability/pipeline/flows/run_compute_epc_weights_flow.py
- asf_heat_pump_suitability/pipeline/flows/run_add_features_flow.py
- asf_heat_pump_suitability/pipeline/flows/run_calculate_garden_size_flow.py
"""

from metaflow import FlowSpec, step, batch, Parameter


class CalculateSuitabilityFlow(FlowSpec):

    # Parameters
    weights = Parameter(
        name="weights",
        help="Path to weighted EPC data, the output of `run_compute_epc_weights.py`",
        required=True,
    )

    epc = Parameter(
        name="epc",
        help="Path to EPC data with added features, the output of `run_add_features.py`",
        required=True,
    )

    gardens = Parameter(
        name="gardens",
        help="Path to deduplicated estimated garden size data, the output of `run_calculate_garden_size.py`.",
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
        Load input datasets and start flow.
        """
        dataset_desc = "sample" if self.sample else "full"
        print(f"Running CalculateSuitabilityFlow on {dataset_desc} EPC dataset.")

        import polars as pl
        from datetime import datetime
        from asf_heat_pump_suitability import config
        from asf_heat_pump_suitability.utils import save_utils

        self.features = config["features"]

        print("Loading EPC data with features")
        self.epc_df = pl.read_parquet(self.epc)
        print("Loading garden size estimates")
        gardens = pl.read_parquet(self.gardens)
        print("Loading weights")
        weights = pl.read_parquet(self.weights)

        print("Joining EPC features data with garden size estimates and weights")
        self.epc_df = self.epc_df.join(gardens, how="left", on="UPRN")
        self.epc_df = self.epc_df.join(weights, how="left", on="UPRN")

        if not self.sample:
            print(f"Saving augmented EPC data")
            save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/augmented_epc/{datetime.today().strftime('%Y%m%d')}_{self.year}_Q{self.quarter}_epc_augmented.parquet"
            save_utils.save_to_s3(self.epc_df, save_as)

        self.next(self.process_features_for_suitability)

    @step
    def process_features_for_suitability(self):
        """
        Process EPC dataset in preparation for calculating suitability.
        """
        import polars as pl
        from asf_heat_pump_suitability.pipeline.suitability import (
            calculate_suitability,
        )

        self.epc_df = self.epc_df.with_columns(
            pl.col("garden_area_m2")
            .fill_null(pl.col("msoa_avg_outdoor_space_m2"))
            .alias("garden_area_m2")
        ).drop("msoa_avg_outdoor_space_m2")

        print("Filtering EPC data to rows with n_features >= minimum threshold")
        self.epc_df = calculate_suitability.filter_df_minimum_features(
            self.epc_df, features=self.features
        )

        if self.sample:
            print("Sampling 1000 rows from EPC data to run pipeline on")
            self.epc_df = self.epc_df.sample(n=1000, seed=2)

        self.next(self.calculate_scores_per_epc_record)

    @step
    def calculate_scores_per_epc_record(self):
        """
        Calculate suitability score per tech type per EPC record.
        """
        from datetime import datetime
        from asf_heat_pump_suitability import config
        from asf_heat_pump_suitability.utils import parallel_utils, save_utils
        from asf_heat_pump_suitability.pipeline.suitability import (
            calculate_suitability,
        )

        tech_types = config["tech_types"]
        scores = []
        for tech_type in tech_types:
            print(f"Calculating suitability scores for tech type: {tech_type}")
            epc_scores_df = calculate_suitability.compute_df_avg_score_per_epc(
                self.epc_df, tech_type
            )
            scores.append(epc_scores_df)

        print("Joining all scores to EPC dataset")
        for score_df in scores:
            self.epc_df = self.epc_df.join(score_df, on="UPRN", how="left")

        if not self.sample:
            save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/suitability/{datetime.today().strftime('%Y%m%d')}_{self.year}_Q{self.quarter}_heat_pump_suitability_per_property.parquet"
            save_utils.save_to_s3(self.epc_df, save_as)

        # Filter to relevant columns
        use_cols = (
            ["lsoa", "proportional_weight"]
            + [col for col in self.epc_df.columns if "score" in col]
            + self.features
        )
        self.epc_df = self.epc_df.select(use_cols)

        if self.sample:
            self.chunks = parallel_utils.chunk_df_by_group(
                self.epc_df,
                group_col="lsoa",
                n=100,
            )
        else:
            # Chunk into dfs of 1000 LSOAs
            self.chunks = parallel_utils.chunk_df_by_group(
                self.epc_df, group_col="lsoa", n=1000
            )

        self.next(self.weight_scores, foreach="chunks")

    @batch(cpu=2, memory=16000)
    @step
    def weight_scores(self):
        """
        Apply weights to scores and aggregate for each LSOA.
        """
        import os

        os.system(
            "pip install git+https://github.com/nestauk/asf_heat_pump_suitability.git"
        )
        from tqdm import tqdm
        from asf_heat_pump_suitability.pipeline.suitability import calculate_suitability

        print("Weighting scores and aggregating per LSOA")
        self.weighted_scores = []

        for lsoa_code in tqdm(self.input["lsoa"].drop_nulls().unique()):
            self.weighted_scores.append(
                calculate_suitability.aggregate_dict_lsoa_suitability_and_features(
                    self.input, lsoa_code
                )
            )

        self.next(self.concatenate_weighted_scores)

    @step
    def concatenate_weighted_scores(self, inputs):
        """
        Concatenate all weighted LSOA suitability scores together and filter to LSOAs with at least 15 records.
        """
        import polars as pl
        import itertools

        self.weighted_scores = list(
            itertools.chain.from_iterable([input.weighted_scores for input in inputs])
        )
        print(
            "Filtering to LSOAs with data for at least 15 properties to be included in final dataset"
        )
        suitability_df = pl.DataFrame(self.weighted_scores).filter(
            pl.col("n_properties") >= 15
        )
        self.suitability_df = suitability_df.with_columns(pl.col(pl.Float64).round(3))

        self.next(self.join_proportion_of_flats_and_names)

    @step
    def join_proportion_of_flats_and_names(self):
        """
        Calculate the proportion of flats in each LSOA from the census data and join it to the suitability scores.
        Join LSOA and DZ names to suitability scores.
        """
        import polars as pl
        from asf_heat_pump_suitability.pipeline.prepare_features import (
            property_type,
            output_areas,
        )

        print("Getting proportion of flats in each LSOA from the census data")
        proportion_flats_df = (
            property_type.transform_df_proportion_census_property_types()
            .filter(pl.col("property_type") == "Flat, maisonette or apartment")
            .select(["lsoa", "census_proportion"])
            .rename({"census_proportion": "census_proportion_flats"})
        )

        print("Getting LSOA & DZ names")
        lsoa_names_df = output_areas.load_df_lsoa_dz_codes_names()

        print("Joining proportion of flats and LSOA & DZ names to suitability dataset")
        self.suitability_df = self.suitability_df.join(
            proportion_flats_df, how="left", on="lsoa"
        ).join(lsoa_names_df, left_on="lsoa", right_on="lsoa_code", how="left")

        self.next(self.save_outputs)

    @step
    def save_outputs(self):
        """
        Save outputs to S3.
        """
        from datetime import datetime
        from asf_heat_pump_suitability.utils import save_utils

        print("Saving LSOA heat pump suitability scores")
        save_as = f"s3://asf-heat-pump-suitability/outputs/{self.year}Q{self.quarter}/suitability/{datetime.today().strftime('%Y%m%d')}_{self.year}_Q{self.quarter}_heat_pump_suitability_per_lsoa"

        if self.sample:
            save_as = save_as + "_SAMPLE"

        save_utils.save_to_s3(self.suitability_df, f"{save_as}.parquet")
        save_utils.save_to_s3(self.suitability_df, f"{save_as}.csv")

        if not self.sample:
            print("Saving open dataset to nesta-open-data S3 bucket")
            save_as = f"s3://nesta-open-data/asf_heat_pump_suitability/{self.year}Q{self.quarter}/{datetime.today().strftime('%Y%m%d')}_{self.year}_Q{self.quarter}_EPC_heat_pump_suitability_per_lsoa"
            save_utils.save_to_s3(self.suitability_df, f"{save_as}.parquet")
            save_utils.save_to_s3(self.suitability_df, f"{save_as}.csv")

        self.next(self.end)

    @step
    def end(self):
        """
        Finish flow.
        """
        print("Calculate suitability flow complete!")


if __name__ == "__main__":
    CalculateSuitabilityFlow()
