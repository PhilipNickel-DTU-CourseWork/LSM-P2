#!/bin/bash
#BSUB -J submit_sequential
#BSUB -o Experiments/cubic/output/submit_cubic_%J.out
#BSUB -e Experiments/cubic/output/submit_cubic_%J.err
#BSUB -n 8
#BSUB -W 00:10
#BSUB -q hpcintro
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

uv sync
mpirun -n 8 uv run python ./Experiments/cubic/cubic.py -N 100 --iter 1000 --output Experiments/cubic/output/result.npy --omega 0.75