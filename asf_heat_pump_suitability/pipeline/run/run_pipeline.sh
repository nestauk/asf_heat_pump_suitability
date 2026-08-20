#!/bin/bash
# This script runs the entire pipeline for a list of local authorities. It executes each step in sequence and checks for errors after each step.
# To make the script executable, run: chmod +x asf_heat_pump_suitability/pipeline/run/run_pipeline.sh
# You can then run the script with: ./asf_heat_pump_suitability/pipeline/run/run_pipeline.sh
# Usage: ./asf_heat_pump_suitability/pipeline/run/run_pipeline.sh [--release_date YYYYMMDD]
# The release date defaults to today and is pinned across all stages, so a run
# crossing midnight still writes to a single dated release directory.

# Run from the repo root regardless of where the script is invoked from
cd "$(dirname "$0")/../../.." || exit 1

# Print why we stopped and how to call this script, then abort.
usage() {
    echo "Error: $1" >&2
    echo "Usage: $0 [--release_date YYYYMMDD]" >&2
    echo "  --release_date  the release date to pin across all stages, e.g. 20260801." >&2
    echo "                  Defaults to today." >&2
    exit 1
}

# Read the arguments: accept an optional --release_date value, reject anything else
release_date=""
while [ $# -gt 0 ]; do
    case "$1" in
        --release_date)
            [ $# -ge 2 ] || usage "--release_date needs a date after it, e.g. --release_date 20260801"
            [ -n "$2" ] || usage "--release_date was given an empty value"
            release_date="$2"
            shift 2
            ;;
        *)
            usage "unknown option '$1'"
            ;;
    esac
done

# Resolve the release date once (defaulting to today) and validate its format
# via get_str_release_date, failing fast before any stage runs
release_date=$(python -c "
import sys
from asf_heat_pump_suitability.utils.save_utils import get_str_release_date
print(get_str_release_date(sys.argv[1] if len(sys.argv) > 1 else None))
" ${release_date:+"$release_date"}) || exit 1
echo "Release date pinned to: $release_date"

local_authorities=(
    "plymouth"
    "dudley"
    "vale of glamorgan"
    "midlothian"
    "glasgow city"
    "south lanarkshire"
    "east lothian"
)

succeeded=0
for la in "${local_authorities[@]}"; do
    echo "=================================================="
    echo "Starting pipeline for: $la"
    echo "=================================================="

    # Step 1: Domestic UPRNs
    echo "--> Running: uprns.py"
    python asf_heat_pump_suitability/pipeline/transform/uprns.py --local_authorities "$la" --release_date "$release_date" --save
    if [ $? -ne 0 ]; then echo "Error in uprns.py for $la. Skipping..."; continue; fi

    # Step 2: Add features
    echo "--> Running: add_features.py"
    python asf_heat_pump_suitability/pipeline/run/add_features.py --local_authorities "$la" --release_date "$release_date" --save
    if [ $? -ne 0 ]; then echo "Error in add_features.py for $la. Skipping..."; continue; fi

    # Step 3: Decision tree
    echo "--> Running: decision_tree.py"
    python asf_heat_pump_suitability/pipeline/transform/decision_tree.py --local_authorities "$la" --release_date "$release_date" --save
    if [ $? -ne 0 ]; then echo "Error in decision_tree.py for $la. Skipping..."; continue; fi

    # Step 4: Clustering / spatial aggregation
    echo "--> Running: cluster.py"
    python asf_heat_pump_suitability/pipeline/cluster/cluster.py --local_authorities "$la" --release_date "$release_date" --save
    if [ $? -ne 0 ]; then echo "Error in cluster.py for $la. Skipping..."; continue; fi

    # Step 5: Compute contextual features
    echo "--> Running: compute_contextual_features.py"
    python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities "$la" --release_date "$release_date" --save
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
