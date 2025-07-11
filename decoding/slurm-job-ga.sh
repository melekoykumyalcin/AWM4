#!/bin/bash
#SBATCH --job-name=tfr_grand
#SBATCH --output=tfr_grand.out
#SBATCH --error=tfr_grand.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=128GBSppc

Load modules and activate environment

module load conda
conda activate neuro-conda-2024a

Run only the grand averaging

python !!! NAME OF THE SCRIPT.PY !!! --grand-average
