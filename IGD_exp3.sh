#!/bin/bash -l

#SBATCH --mail-user=nick.lemke@gris.informatik.tu-darmstadt.de
#SBATCH -J nnNCA
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


CUDA_VISIBLE_DEVICES=0 nnUNet_train_nca 3d_fullres -t 506 -f 0 -num_epoch 250 -d 0 -save_interval 25 -s seg_outputs --store_csv --fp32