#!/bin/bash

# Setup script to configure SLURM job scripts with account and partition information

echo "=========================================="
echo "SLURM Job Script Configuration"
echo "=========================================="
echo ""

# Check if user provided account and partition
if [ $# -lt 2 ]; then
    echo "Usage: $0 <account> <partition>"
    echo ""
    echo "Example: $0 EUHPC_D03_006 boost_usr_prod"
    echo ""
    echo "To find your account, run: sacctmgr show associations user=\$USER format=account"
    echo "To find available partitions, run: sinfo -o '%P'"
    echo ""
    exit 1
fi

ACCOUNT=$1
PARTITION=$2

echo "Account: $ACCOUNT"
echo "Partition: $PARTITION"
echo ""

# List of job scripts to configure
SCRIPTS=(
    "mobilenet_v1_deduplicate.sh"
    "mobilenet_v2_deduplicate.sh"
    "shufflenet_deduplicate.sh"
    "squeezenet_deduplicate.sh"
)

echo "Configuring job scripts..."
echo ""

for SCRIPT in "${SCRIPTS[@]}"; do
    if [ -f "$SCRIPT" ]; then
        echo "  Configuring $SCRIPT..."
        sed -i "s/REPLACE_WITH_YOUR_ACCOUNT/$ACCOUNT/g" "$SCRIPT"
        sed -i "s/REPLACE_WITH_PARTITION/$PARTITION/g" "$SCRIPT"
        chmod +x "$SCRIPT"
        echo "    ✓ Done"
    else
        echo "    ✗ Warning: $SCRIPT not found"
    fi
done

echo ""
echo "=========================================="
echo "Configuration Complete"
echo "=========================================="
echo ""
echo "To submit the jobs, run:"
echo ""
for SCRIPT in "${SCRIPTS[@]}"; do
    echo "  sbatch $SCRIPT"
done
echo ""
echo "To monitor your jobs, use:"
echo "  squeue -u \$USER"
echo ""
echo "To view job output while running:"
echo "  tail -f <output_file>.out"
echo ""
