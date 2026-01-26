# Layer Deduplication Job Scripts

This directory contains scripts to deduplicate layers across MobileNet, ShuffleNet, and SqueezeNet models using the Leonardo HPC cluster scheduler (SLURM).

## Files

- `find_remove_duplicates.py` - Main deduplication script
- `compare_layers.py` - Utility to compare two layers
- `setup_env.sh` - Configuration helper script
- `mobilenet_v1_deduplicate.sh` - SLURM job for MobileNet v1
- `mobilenet_v2_deduplicate.sh` - SLURM job for MobileNet v2
- `shufflenet_deduplicate.sh` - SLURM job for ShuffleNet v1 & v2
- `squeezenet_deduplicate.sh` - SLURM job for SqueezeNet

## Setup Instructions

### 1. Find Your SLURM Account and Partition

```bash
# Find your account
sacctmgr show associations user=$USER format=account

# Find available partitions
sinfo -o '%P'
```

### 2. Configure Job Scripts

Run the setup script with your account and partition:

```bash
cd /leonardo/home/userexternal/cdiaconu/Deeploy
bash setup_env.sh <your_account> <your_partition>
```

Example:
```bash
bash setup_env.sh EUHPC_D03_006 boost_usr_prod
```

### 3. Submit Jobs

Submit all jobs:
```bash
sbatch mobilenet_v1_deduplicate.sh
sbatch mobilenet_v2_deduplicate.sh
sbatch shufflenet_deduplicate.sh
sbatch squeezenet_deduplicate.sh
```

Or submit them one at a time as needed.

### 4. Monitor Jobs

Check job status:
```bash
squeue -u $USER
```

View job output in real-time:
```bash
tail -f mobilenet_v1_dedup_<job_id>.out
```

Cancel a job if needed:
```bash
scancel <job_id>
```

## Job Details

Each job will:
- Process multiple model versions in sequence
- Create a `layer_deduplication_report.md` in each model's `Layers/` directory
- Remove duplicate layers (keeping only one instance of each unique configuration)
- Log output to `*_dedup_<job_id>.out` and errors to `*_dedup_<job_id>.err`

### Resource Allocation

Each job requests:
- 1 node
- 1 CPU core
- 4GB RAM
- 30 minutes wall time

## Models Processed

### MobileNet v1
- x0_25, x0_5, x0_75, x1_0

### MobileNet v2
- x0_5, x0_75, x1_0, x1_4
- (x0_35 already processed manually, OldModel excluded)

### ShuffleNet v1 & v2
- v1: x0_5, x1_0, x1_5, x2_0
- v2: x0_5, x1_0, x1_5, x2_0

### SqueezeNet
- v1_0, v1_1

## Output

Each processed model will have a `layer_deduplication_report.md` file in its `Layers/` directory containing a table with:
- Layer Name
- Layer Type
- Occurrences (number of duplicates found)

## Troubleshooting

If a job fails:
1. Check the error file: `*_dedup_<job_id>.err`
2. Check the output file: `*_dedup_<job_id>.out`
3. Verify the directories exist: `ls DeeployTest/Tests/Models/<model>/<version>/Layers`
4. Test manually: `python3 find_remove_duplicates.py <path_to_layers> --dry-run`

## Manual Execution

To run without the scheduler (for testing):
```bash
python3 find_remove_duplicates.py DeeployTest/Tests/Models/MobileNet/v1/x0_25/Layers --dry-run
```

Remove `--dry-run` to actually delete duplicates.
