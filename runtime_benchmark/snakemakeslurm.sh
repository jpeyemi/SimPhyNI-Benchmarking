#!/bin/bash
# Runtime Benchmark — Snakemake SLURM submission script
# Mirrors the root snakemakeslurm.sh for the runtime_benchmark sub-workflow.
#
# Usage (from runtime_benchmark/ directory):
#   bash snakemakeslurm.sh              # full run
#   bash snakemakeslurm.sh -n           # dry run
#   bash snakemakeslurm.sh --until run_simphyni   # partial run

SM_PARAMS="job-name ntasks partition time mail-user mail-type error output"
SM_ARGS=" --no-requeue --parsable --cpus-per-task {cluster.cpus-per-task} --mem {cluster.mem}"
for P in ${SM_PARAMS}; do SM_ARGS="$SM_ARGS --$P {cluster.$P}"; done

# logs/ dir expected by cluster.slurm.json error/output paths
mkdir -p logs

conda config --set ssl_verify no

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

snakemake -p \
    $* \
    --latency-wait 120 \
    -j 75 \
    --cluster-config "${SCRIPT_DIR}/cluster.slurm.json" \
    --cluster "sbatch $SM_ARGS" \
    --cluster-status /home/iobal/mit_lieberman/scripts/slurm_status.py \
    --rerun-incomplete \
    --restart-times 2 \
    --keep-going \
    --use-conda \
    --conda-frontend conda \
    --conda-prefix /home/iobal/mit_lieberman/tools/conda_snakemake \
    -s "${SCRIPT_DIR}/Snakefile" \
    --rerun-triggers mtime
