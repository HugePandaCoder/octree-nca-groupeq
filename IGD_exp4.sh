#!/bin/bash -l

#SBATCH --mail-user=john.kalkhof@gris.tu-darmstadt.de
#SBATCH -J AAAIEXP4
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8192 
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /gris/gris-f/homestud/jkalkhof/SLURM/logs/%j_%J.out
#SBATCH -e /gris/gris-f/homestud/jkalkhof/SLURM/logs/%j_%J.err

eval "$(/gris/gris-f/homestud/jkalkhof/anaconda3/bin/conda shell.bash hook)"
cd /gris/gris-f/homestud/jkalkhof/projects/NCA
conda activate NCA_310
export PYTHONPATH=.

python /gris/gris-f/homestud/jkalkhof/projects/NCA/train_Diffusion_NCA_IGD4.py