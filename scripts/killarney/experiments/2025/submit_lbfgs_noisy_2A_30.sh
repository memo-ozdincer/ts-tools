#!/bin/bash
set -euo pipefail

# Submits the two isolated L-BFGS precondition experiments (HIP + SCINE), 30 samples each, 2.0Å noise.
# Usage:
#   bash scripts/killarney/experiments/submit_lbfgs_noisy_2A_30.sh

sbatch scripts/killarney/experiments/scine_lbfgs_minimize_noisy.slurm
sbatch scripts/killarney/experiments/hip_lbfgs_minimize_noisy.slurm
