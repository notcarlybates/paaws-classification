#!/bin/bash
#SBATCH --job-name=handedness_eval
#SBATCH --output=/scratch/bates.car/jobs/%j/job_%j.out
#SBATCH --error=/scratch/bates.car/jobs/%j/job_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=short

cd /home/bates.car/testing/paaws-classification

source .venv/bin/activate

python compare_handedness.py
