#!/bin/bash
#SBATCH --job-name=mobilenet_v2_dedup
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=mobilenet_v2_dedup_%j.out
#SBATCH --error=mobilenet_v2_dedup_%j.err
#SBATCH --account=iscrc_demollm
#SBATCH --partition=lrd_all_serial

# MobileNet v2 Layer Deduplication Job (excluding OldModel and x0_35 already done)

set -e

SCRIPT_DIR="/leonardo/home/userexternal/cdiaconu/Deeploy"
DEDUP_SCRIPT="$SCRIPT_DIR/find_remove_duplicates.py"

# MobileNet v2 directories (excluding OldModel and x0_35)
DIRS=(
    "$SCRIPT_DIR/DeeployTest/Tests/Models/MobileNet/v2/x0_5/Layers"
    "$SCRIPT_DIR/DeeployTest/Tests/Models/MobileNet/v2/x0_75/Layers"
    "$SCRIPT_DIR/DeeployTest/Tests/Models/MobileNet/v2/x1_0/Layers"
    "$SCRIPT_DIR/DeeployTest/Tests/Models/MobileNet/v2/x1_4/Layers"
)

echo "=========================================="
echo "MobileNet v2 Layer Deduplication"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

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
echo "MobileNet v2 Processing Complete"
echo "=========================================="
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Total: ${#DIRS[@]}"
echo "End time: $(date)"
echo ""
