#!/bin/bash -l

#SBATCH --mail-user=nick.lemke@gris.informatik.tu-darmstadt.de
#SBATCH -J nnNCA_eval
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=8192 
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /gris/gris-f/homestud/nlemke/SLURM/logs/%j_%J.out
#SBATCH -e /gris/gris-f/homestud/nlemke/SLURM/logs/%j_%J.err

eval "$(/gris/gris-f/homestud/nlemke/miniconda3/bin/conda shell.bash hook)"
cd /gris/gris-f/homestud/nlemke
source configFiler

CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate2 3d_fullres nnUNetTrainerNCA -trained_on 11 -f 0 -use_model 11 -evaluate_on 11 12 13 15 16 --store_csv -d 0 --fp32