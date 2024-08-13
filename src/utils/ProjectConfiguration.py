import os
class ProjectConfiguration:
    STUDY_PATH = r"/local/scratch/clmn1/octree_study/"
    VITB16_WEIGHTS = r"/home/jkalkhof_locale/Documents/GitHub/PretrainedVITs/imagenet21k_R50+ViT-B_16.npz"

if not os.path.exists(ProjectConfiguration.STUDY_PATH):
    ProjectConfiguration.STUDY_PATH = ProjectConfiguration.STUDY_PATH.replace(r"/local/scratch/", r"/gris/scratch-gris-filesrv/")