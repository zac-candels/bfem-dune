#!/bin/bash
#SBATCH --job-name=dropSpread
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --account=eidf266
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

SIF="/epccfs/eidf266/eidf266/s2767572-eidf266/dune.sif"
WORKDIR="$SLURM_SUBMIT_DIR"

echo "Job directory: $WORKDIR"
echo "Container image: $SIF"

if [ ! -r "$SIF" ]; then
    echo "ERROR: Container image is not accessible from this compute node."
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
        exec /opt/dune-env/bin/python dropSpread.py
    '
