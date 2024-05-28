# Med-NCA, M3D-NCA
 
# Fid Score Calculation
Create FID Model from dataset that can be loaded:
python -m pytorch_fid --save-stats path/to/dataset path/to/outputfile

python -m pytorch_fid --save-stats path/to/dataset /home/jkalkhof_locale/Documents/GitHub/NCA/Study/Datasets_FID



Installation
conda create -n nca3 python=3.10.14
DS_BUILD_CPU_ADAM=1 pip install deepspeed