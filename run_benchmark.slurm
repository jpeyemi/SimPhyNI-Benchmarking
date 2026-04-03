#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH -p pi_tami
#SBATCH -n 1
#SBATCH --time=72:00:00
#SBATCH -o mainout.txt
#SBATCH -e mainerr.txt
#SBATCH --mem=16000
#SBATCH --mail-user=iobal@mit.edu
#SBATCH --mail-type=FAIL

bash snakemakeslurm.sh

echo Done!!!
