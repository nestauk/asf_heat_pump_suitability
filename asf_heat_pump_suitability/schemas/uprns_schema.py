import pandera as pa

# Validate EPC UPRNs are max 12 digit positive integers
EPC_UPRN_Schema = pa.DataFrameSchema(
    {"UPRN": pa.Column("Int64", pa.Check.in_range(1, 999999999999), nullable=True)},
    strict=False,
)


def create_domestic_uprn_schema(
    min_expected_rows: int, max_expected_rows: int
) -> pa.DataFrameSchema:
    """
    Validate all columns of final filtered domestic UPRN dataframe prior to saving.

    Performs the following checks on the columns:
    UPRN: integer, unique, maximum 12 digit positive number, exists for all entries
    X_COORDINATE: float, within expected geographical bounds (BNG), exists for all entries
    Y_COORDINATE: float, within expected geographical bounds (BNG), exists for all entries
    LATITUDE: float, within expected geographical bounds
    LONGITUDE: float, within expected geographical bounds
    LAD23CD: string
    LAD23NM: string

    Performs the following checks on the dataframe:
    Number of rows within an expected range

    Args:
        min_expected_rows (int): minimum expected UPRNs in the final dataset
        max_expected_rows (int): maximum expected UPRNs in the final dataset

    Returns:
        schema for validating domestic UPRN dataframe
    """

    Domestic_UPRN_Schema = pa.DataFrameSchema(
        columns={
            "UPRN": pa.Column(
                "Int64",
                unique=True,
                checks=pa.Check.in_range(1, 999999999999),
                nullable=False,
            ),
            "X_COORDINATE": pa.Column(
                float, checks=pa.Check.in_range(0.0, 700000.0), nullable=False
            ),
            "Y_COORDINATE": pa.Column(
                float, checks=pa.Check.in_range(0.0, 1300000.0), nullable=False
            ),
            "LATITUDE": pa.Column(
                float, checks=pa.Check.in_range(49.0, 61.0), nullable=True
            ),
            "LONGITUDE": pa.Column(
                float, checks=pa.Check.in_range(-9.0, 2.0), nullable=True
            ),
            "LAD23CD": pa.Column(str, nullable=True),
            "LAD23NM": pa.Column(str, nullable=True),
        },
        checks=[
            pa.Check(
                lambda df: (min_expected_rows <= len(df) <= max_expected_rows),
                name="uprn_census_household_comparison",
                ignore_na=False,
            )
        ],
        strict=True,
    )

    return Domestic_UPRN_Schema
