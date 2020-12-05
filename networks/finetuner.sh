#!/bin/bash

#job name
#SBATCH -J mlevy_1

#number of nodes
#SBATCH -N 1

#walltime (set to 24 hours)
#SBATCH -t 1-12:00:00

#memory size
#SBATCH --mem=64gb

#SBATCH --output outputs/run_output.%j 
#SBATCH --qos=medium
#SBATCH --partition=dpart
#SBATCH --gres=gpu:1

CUDA_VISIBLE_DEVICES=0
srun python -u starter.py

