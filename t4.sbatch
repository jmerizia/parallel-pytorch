#!/bin/bash

# auto-generated script DO NOT EDIT MANUALLY

#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --time 00:10:00
#SBATCH --partition t4_normal_q
#SBATCH --account distdl
#SBATCH --output output.log
#SBATCH --gres gpu:1

set -e

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Load modules
module reset
module load SciPy-bundle/2020.11-fosscuda-2020b
module load PyTorch/1.7.1-fosscuda-2020b
module load mpi4py/3.1.1-gompi-2020b-timed-pingpong
#pip install -e ~/git/distdl

mpirun --n 8 python -B -m mpi4py benchmark.py

