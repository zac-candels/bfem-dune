#!/bin/bash
#SBATCH --job-name=dropSpread
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --account=eidf266
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Require two arguments: tau and theta
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch submit.slurm TAU THETA"
    exit 1
fi

TAU="$1"
THETA="$2"

SIF="/epccfs/eidf266/eidf266/s2767572-eidf266/dune.sif"
WORKDIR="$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Job directory: $WORKDIR"
echo "Relaxation time: $TAU"
echo "Contact angle: $THETA"
echo "Container image: $SIF"

if [ ! -r "$SIF" ]; then
    echo "ERROR: Container image is not accessible."
    exit 1
fi

mkdir -p "$WORKDIR/.cache/dune" \
         "$WORKDIR/.cache/matplotlib"

 /usr/bin/apptainer exec \
    --bind "$WORKDIR:/workspace" \
    --bind "$WORKDIR/.cache:/host-cache" \
    "$SIF" \
    bash -c '
        export DUNE_PY_DIR=/host-cache/dune
        export XDG_CACHE_HOME=/host-cache
        export MPLCONFIGDIR=/host-cache/matplotlib

        cd /workspace

        exec /opt/dune-env/bin/python dropSpreadAdaptiveRefinement.py \
            --tau "$1" \
            --theta "$2"
    ' _ "$TAU" "$THETA"