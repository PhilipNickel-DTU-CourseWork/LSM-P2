#!/bin/bash
#BSUB -J submit_sequential
#BSUB -o Experiments/template/output/submit_sequential_%J.out
#BSUB -e Experiments/template/output/submit_sequential_%J.err
#BSUB -n 1
#BSUB -W 00:05
#BSUB -q hpcintro
#BSUB -R "rusage[mem=8GB]"

module load python3/3.11.1

uv sync
uv run python ./Experiments/template/sequential.py -N 200 --iter 500 --tolerance 1e-8 --plot_convergence Experiments/template/output