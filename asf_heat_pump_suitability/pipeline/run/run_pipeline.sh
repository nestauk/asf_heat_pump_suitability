#!/bin/bash
# This script runs the entire pipeline for a list of local authorities. It executes each step in sequence and checks for errors after each step.
# To make the script executable, run: chmod +x asf_heat_pump_suitability/pipeline/run/run_pipeline.sh
# You can then run the script with: ./asf_heat_pump_suitability/pipeline/run/run_pipeline.sh

# Run from the repo root regardless of where the script is invoked from
cd "$(dirname "$0")/../../.." || exit 1

# Preflight: verify every configured S3 input path exists before starting
echo "--> Checking S3 input paths exist: check_inputs.py"
python asf_heat_pump_suitability/pipeline/validate/check_inputs.py
if [ $? -ne 0 ]; then
    echo "Error running check_inputs.py: missing S3 input paths. Aborting."
    exit 1
fi

local_authorities=(
    "plymouth"
    "dudley"
    "vale of glamorgan"
    "midlothian"
    "glasgow city"
    "south lanarkshire"
    "east lothian"
)

succeeded = 0
for la in "${local_authorities[@]}"; do
    echo "=================================================="
    echo "Starting pipeline for: $la"
    echo "=================================================="

    # Step 1: Domestic UPRNs
    echo "--> Running: uprns.py"
    python asf_heat_pump_suitability/pipeline/transform/uprns.py --local_authorities "$la" --save
    if [ $? -ne 0 ]; then echo "Error in uprns.py for $la. Skipping..."; continue; fi

    # Step 2: Add features
    echo "--> Running: add_features.py"
    python asf_heat_pump_suitability/pipeline/run/add_features.py --local_authorities "$la" --save
    if [ $? -ne 0 ]; then echo "Error in add_features.py for $la. Skipping..."; continue; fi

    # Step 3: Decision tree
    echo "--> Running: decision_tree.py"
    python asf_heat_pump_suitability/pipeline/transform/decision_tree.py --local_authorities "$la" --save
    if [ $? -ne 0 ]; then echo "Error in decision_tree.py for $la. Skipping..."; continue; fi

    # Step 4: Clustering / spatial aggregation
    echo "--> Running: cluster.py"
    python asf_heat_pump_suitability/pipeline/cluster/cluster.py --local_authorities "$la" --save
    if [ $? -ne 0 ]; then echo "Error in cluster.py for $la. Skipping..."; continue; fi

    # Step 5: Compute contextual features
    echo "--> Running: compute_contextual_features.py"
    python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities "$la" --save
    if [ $? -ne 0 ]; then echo "Error in compute_contextual_features.py for $la. Skipping..."; continue; fi

    echo "Successfully finished pipeline for: $la"
    succeeded=$((succeeded+1))
    echo ""
done

echo "Pipeline completed for ${succeeded:-0} of ${#local_authorities[@]} local authorities."
echo "=================================================="

echo "--> Generating manifest.json: create_manifest.py"
python asf_heat_pump_suitability/pipeline/run/create_manifest.py
if [ $? -ne 0 ]; then
    echo "Error running create_manifest.py"
    exit 1
fi

echo "Pipeline finished!"
