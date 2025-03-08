import pandas as pd
import os

# Load the dataset (adjust the file name/path as needed)
# df = pd.read_csv("/Users/aidan.kelly/nesta/ASF/asf_heat_pump_suitability/outputs")
# df = pd.read_csv('../../../outputs/hn_zones/output_data/la_mae_data.csv')

from asf_heat_pump_suitability import PROJECT_DIR

INPUT_DATA_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/output_data/")

input_file_path = os.path.join(INPUT_DATA_DIR, "la_mae_data.csv")

df = pd.read_csv(
    "/Users/aidan.kelly/nesta/ASF/asf_heat_pump_suitability/outputs/hn_zones/output_data/la_mae_data.csv"
)

# Sort by 'avg_hn_score_pilot_nonzero' in ascending order
df_sorted = df.sort_values(by="avg_hn_score_pilot_nonzero", ascending=True)

# Reshape the dataset from wide to long format
reshaped_df = pd.melt(
    df_sorted,
    id_vars=["Local Authority"],  # Keep LAs as identifiers
    value_vars=[
        "avg_hn_score_pilot_zero",
        "avg_hn_score",
        "avg_hn_score_pilot_nonzero",
    ],
    var_name="Score Type",  # Equivalent to the "Year" column in your example
    value_name="Score",  # Equivalent to "Life expectancy"
)

# Rename the score type values to more descriptive names
rename_dict = {
    "avg_hn_score_pilot_zero": "Average HN score for when a DESNZ HN zone is absent",
    "avg_hn_score": "Average HN score for all",
    "avg_hn_score_pilot_nonzero": "Average HN score for when a DESNZ HN zone is present",
}
reshaped_df["Score Type"] = reshaped_df["Score Type"].replace(rename_dict)

# Ensure LAs maintain the same order for all score types
reshaped_df["Local Authority"] = pd.Categorical(
    reshaped_df["Local Authority"],
    categories=df_sorted["Local Authority"].unique(),
    ordered=True,
)

# Sort to maintain the order in the reshaped dataframe
reshaped_df = reshaped_df.sort_values(by=["Local Authority", "Score Type"])


print("\nReshaped Data:")
print(reshaped_df.head(10))  # show first 10 rows

# Define the output directory
OUTPUT_DATA_DIR = os.path.join(PROJECT_DIR, "outputs/hn_zones/flourish_prepped_data/")

# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# Save the reshaped dataframe to a CSV file
output_file_path = os.path.join(OUTPUT_DATA_DIR, "la_mae_reshaped_data.csv")

reshaped_df.to_csv(output_file_path, index=False)
