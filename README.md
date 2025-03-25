# OctreeNCA

This is the source code of our OctreeNCA paper, currently under review. Confidential, do not distribute!

## Setup and installation

1. Create a conda environment with ``conda create -n nca python=3.10.14``
2. Set a ``FILER_BASE_PATH`` and ``STUDY_PATH`` in [`src/utils/ProjectConfiguration.py`](src/utils/ProjectConfiguration.py). The results of the training will be stored at ``os.path.join(FILER_BASE_PATH, STUDY_PATH)``
3. Install dependencies (by trial-and-error). Refrain from installing ``segment_anything`` unless necessary.

## Visualize and Debug Experiments with Aim Logger
1. Navigate to the output folder specified in [`src/utils/ProjectConfiguration.py`](src/utils/ProjectConfiguration.py).
2. Enter the subfolder ``Aim``. 
3. Run ``aim up``.
4. Open the displayed URL in your browser to view the training results.

## OctreeNCA implementation

The implementation of our OctreeNCA can be found in [`src/models/Model_OctreeNCA_2d_patching2.py`](src/models/Model_OctreeNCA_2d_patching2.py) and [`src/models/Model_OctreeNCA_3d_patching2.py`](src/models/Model_OctreeNCA_3d_patching2.py)

## CUDA Inference Functions

The CUDA inference functions proposed in our paper can be found in [`nca_cuda.cu`](nca_cuda.cu) and [`nca_cuda3d.cu`](nca_cuda3d.cu). The extensions are compiled with the [`nca_cuda_compile.py`](nca_cuda_compile.py) script. In our case we need to specify the path to the CUDA toolkit installation:

```bash 
CUDA_HOME=/usr/local/cuda-12.4 python nca_cuda_compile.py install
```

## Preprocessing
The radiology and pathology data does not require any preprocessing. 
The surgical videos are preprocessed using the [`preprocess_cholecSeg8k.ipynb`](preprocess_cholecSeg8k.ipynb) notebook.


## Scripts for Creating the Figures


| | Content | Main script | Additional scripts (e.g. for  training) |
|-|-|-|-|
|1|VRAM|``figure_vram_inference.ipynb``|``inference_pathology_unet_gpu_vram.py``, ``inference_pathology_oct_gpu_vram.py``, ``inference_pathology_m3d_gpu_vram.py``, ``test_segformer.py, test_transunet.py``|
|5|Pathology|``figure_peso.ipynb``|``train_peso_med.py``, ``train_peso_min_unet.py``, ``train_peso_sam.py``, ``train_peso_segformer.py``, ``train_peso_unet.py``, ``train_peso.py``|
|6|Surgery|``figure_cholec.ipynb``|``train_cholecSeg.py``, ``train_cholecSeg_unet.py``, ``train_cholecSeg_sam2.py``, ``train_cholecSeg_min_unet_3d.py``, ``train_cholecSeg_m3d.py``|
|7|Radiology|``figure_prostate.ipynb``|``train_prostate2.py``, ``train_prostate_unet_3d.py``, ``train_prostate_min_unet_3d.py``, ``train_prostate_m3d.py``|
|8|Pathology Qualitative|``eval_peso.ipynb``|``patchwise_inference_peso_unet.py``, ``train_peso_unet.py``, ``train_peso.py``|
|9|Raspberry Pi|``figure_raspi_runtime.ipynb``|``pi_measure_med.py``, ``pi_measure_oct.py``, ``pi_measure_unet.py``|
|10|CUDA vs. traditional|``figure_single_nca_step.ipynb``|-|
|11|Function vs. local|``inference2_cholec_oct.py``|-|
|12|Radiology Qualitative|``figure_qualitative_prostate.ipynb``|``train_prostate2.py``, ``train_prostate_unet_3d.py``, ``train_prostate_min_unet_3d.py``, ``train_prostate_m3d.py``|
|13|Pathology Qualitative|``eval_peso.ipynb``|``patchwise_inference_peso_unet.py``, ``inference2_peso_oct.py``|
|14|Surgery Qualitative|``figure_qualitative_cholec.ipynb``|``train_cholecSeg.py``, ``train_cholecSeg_unet.py``, ``train_cholecSeg_m3d.py``|

## Scripts for Creating the Tables

| | Content | Main script | Additional scripts (e.g. for  training) |
|-|-|-|-|
|1|Ablations|``table_ablation.ipynb``|``train_prostate2.py``, ``train_cholecSeg.py``, ``train_peso.py``|
|2|Large-scale inference|-|``inference2_cholec_m3d.py``, ``inference2_cholec_oct.py``, ``inference2_cholec_unet.py``, ``inference2_peso_m3d.py``, ``inference2_peso_oct.py``, ``inference2_peso_unet.py``|
|4|Radiology|``table_prostate.ipynb``|``train_prostate2.py``, ``train_prostate_unet_3d.py``, ``train_prostate_min_unet_3d.py``, ``train_prostate_m3d.py``|
|5|Pathology|``table_peso.ipynb``|``train_peso_med.py``, ``train_peso_min_unet.py``, ``train_peso_sam.py``, ``train_peso_segformer.py``, ``train_peso_unet.py``, ``train_peso.py``|
|6|Surgery|``table_cholec.ipynb``|``train_cholecSeg.py``, ``train_cholecSeg_unet.py``, ``train_cholecSeg_sam2.py``, ``train_cholecSeg_min_unet_3d.py``, ``train_cholecSeg_m3d.py``|



## Notes for myself
Installation
conda create -n nca3 python=3.10.14
DS_BUILD_CPU_ADAM=1 pip install deepspeed


git push -u github octree_nca