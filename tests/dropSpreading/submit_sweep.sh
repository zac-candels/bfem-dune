#!/bin/bash

set -euo pipefail

TAUS=(0.001 0.01 0.1 1 2)
ANGLES=( $(seq 30 15 150) )

for tau in "${TAUS[@]}"; do
    for theta in "${ANGLES[@]}"; do
        echo "Submitting tau=$tau, theta=$theta"
        sbatch run_dune.slurm "$tau" "$theta"
    done
done