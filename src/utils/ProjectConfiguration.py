import os
class ProjectConfiguration:
    STUDY_PATH = r"pvc_groupEq/baseline_results/"
    VITB16_WEIGHTS = r"/home/jkalkhof_locale/Documents/GitHub/PretrainedVITs/imagenet21k_R50+ViT-B_16.npz"
    FILER_BASE_PATH = r"/local/scratch/"

if not os.path.exists(ProjectConfiguration.FILER_BASE_PATH):
    ProjectConfiguration.FILER_BASE_PATH = r"/gris/scratch-gris-filesrv/"