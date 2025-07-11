#!/bin/bash

#SBATCH --job-name=tfr_sub_%a

#SBATCH --output=tfr_sub_%a.out

#SBATCH --error=tfr_sub_%a.err

#SBATCH --time=3:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=8

#SBATCH --partition=128GBSppc

#SBATCH --array=1-30  

# Load modules and activate environment

module load conda
conda activate neuro-conda-2024a

*# Process single subject*

python !!!!NAME OF THE SCRIPT.PY !!!!! --subject ${SLURM_ARRAY_TASK_ID}
