#!/bin/bash -l

#SBATCH --mail-user=nick.lemke@gris.informatik.tu-darmstadt.de
#SBATCH -J octree
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8192 
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /gris/gris-f/homestud/nlemke/SLURM/logs/%j_%J.out
#SBATCH -e /gris/gris-f/homestud/nlemke/SLURM/logs/%j_%J.err

eval "$(/gris/gris-f/homestud/nlemke/miniconda3/bin/conda shell.bash hook)"
cd /gris/gris-f/homestud/nlemke/NCA
conda activate nca3
export PYTHONPATH=.

python /gris/gris-f/homestud/nlemke/NCA/train_cholecSeg_unet.py