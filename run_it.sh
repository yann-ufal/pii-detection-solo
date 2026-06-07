#!/bin/bash

# This is a Slurm batch script for PII inference across multi LUMI-G nodes
#
# Submit the job with: sbatch run_it.sh
#
# One LUMI-G node = 8 GCDs
# ignite one torchrun per node (srun --ntasks-per-node=1)
#
# Multi-node rank assignment: torchrun runs --standalone (per-node, local-only rendezvous)
# No flaky cross-node rendezvous barrier


#SBATCH --job-name=pii-samples             # <-- EDIT to be unique
#SBATCH --account=project_465002530      # <-- EDIT
#SBATCH --partition=standard-g           # full LUMI-G nodes (required for multi-node)
#SBATCH --nodes=1                       # <-- EDIT to fit data volume
#SBATCH --ntasks-per-node=1              # one torchrun per node - fans out to 8 procs
#SBATCH --gpus-per-node=8                
#SBATCH --cpus-per-task=56               
#SBATCH --mem=480G                       # near the 512G per-node maximum
#SBATCH --time=12:00:00                  # <-- EDIT to fit data volume
#SBATCH --output=logs/pii-%j.out
#SBATCH --error=logs/pii-%j.err

set -euo pipefail
mkdir -p logs

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
SQUASHFS_NAME="$PWD/user-software.sqsh"

# Slingshot + RCCL + MPI bindings + working-dir access into containers
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Make the upgraded transformers (from the .sqsh) visible inside the container by prepending the venv's bin to PATH
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# Use the model cache baked into the .sqsh. Force fully offline operation
export SINGULARITYENV_HF_HOME=/user-software/hf
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1

# Logs every torch.compile recompilation
# export SINGULARITYENV_TORCH_LOGS=recompiles

# Per-job MIOpen scratch dirs
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR
export SINGULARITYENV_MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB


srun --ntasks-per-node=1 singularity run \
    --overlay "${SQUASHFS_NAME}:ro" \
    "$SIF" \
    bash -c 'export PYTHONPATH=/user-software/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}; \
        python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        pii_infer.py'

# Clean up
rm -rf "$MIOPEN_DIR"
