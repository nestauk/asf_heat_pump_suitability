#!/bin/bash

local_authorities=(
    "plymouth"
    "dudley"
    "vale of glamorgan"
    "midlothian"
    "glasgow city"
    "south lanarkshire"
    "east lothian"
)

for la in "${local_authorities[@]}"; do
    echo "=================================================="
    echo "Starting pipeline for: ${la^^}"
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
    echo ""
done

echo "All local authorities data generated!"
echo "=================================================="

echo "--> Generating manifest.json: create_manifest.py"
python asf_heat_pump_suitability/pipeline/run/create_manifest.py
if [ $? -ne 0 ]; then
    echo "Error running create_manifest.py"
    exit 1
fi

echo "Pipeline finished!"
