#!/bin/bash

# Batch deduplication script for MobileNet, ShuffleNet, and SqueezeNet models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEDUP_SCRIPT="$SCRIPT_DIR/find_remove_duplicates.py"

# Array of all directories to process
DIRS=(
    # MobileNet v1
    "DeeployTest/Tests/Models/MobileNet/v1/x0_25/Layers"
    "DeeployTest/Tests/Models/MobileNet/v1/x0_5/Layers"
    "DeeployTest/Tests/Models/MobileNet/v1/x0_75/Layers"
    "DeeployTest/Tests/Models/MobileNet/v1/x1_0/Layers"

    # MobileNet v2 (excluding OldModel and x0_35 which is already done)
    "DeeployTest/Tests/Models/MobileNet/v2/x0_5/Layers"
    "DeeployTest/Tests/Models/MobileNet/v2/x0_75/Layers"
    "DeeployTest/Tests/Models/MobileNet/v2/x1_0/Layers"
    "DeeployTest/Tests/Models/MobileNet/v2/x1_4/Layers"

    # ShuffleNet v1
    "DeeployTest/Tests/Models/ShuffleNet/v1/x0_5/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v1/x1_0/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v1/x1_5/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v1/x2_0/Layers"

    # ShuffleNet v2
    "DeeployTest/Tests/Models/ShuffleNet/v2/x0_5/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v2/x1_0/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v2/x1_5/Layers"
    "DeeployTest/Tests/Models/ShuffleNet/v2/x2_0/Layers"

    # SqueezeNet
    "DeeployTest/Tests/Models/SqueezeNet/v1_0/Layers"
    "DeeployTest/Tests/Models/SqueezeNet/v1_1/Layers"
)

echo "=========================================="
echo "Batch Layer Deduplication"
echo "=========================================="
echo "Total directories to process: ${#DIRS[@]}"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_REMOVED=0
TOTAL_KEPT=0

for DIR in "${DIRS[@]}"; do
    echo "=========================================="
    echo "Processing: $DIR"
    echo "=========================================="

    if [ ! -d "$DIR" ]; then
        echo "WARNING: Directory does not exist, skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        continue
    fi

    # Run the deduplication script
    if python3 "$DEDUP_SCRIPT" "$DIR"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ Successfully processed $DIR"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ Failed to process $DIR"
    fi

    echo ""
done

echo "=========================================="
echo "Batch Processing Complete"
echo "=========================================="
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Total: ${#DIRS[@]}"
echo ""
