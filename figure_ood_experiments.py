
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


#octree_nca = r"/local/scratch/clmn1/octree_study/Experiments/new_prostate_overkill5_OctreeNCASegmentation"
octree_nca = r"/local/scratch/clmn1/octree_study/Experiments/new_prostate_overkill6_OctreeNCASegmentation"
#octree_nca = r"/local/scratch/clmn1/octree_study/Experiments/new_prostate_copied_OctreeNCASegmentation"
m3d_nca = r"/local/scratch/clmn1/octree_study/Experiments/new_prostate_m3d_M3dSegmentation"
unet = r"/local/scratch/clmn1/octree_study/Experiments/new_prostate_unet_3d_8_UNetSegmentation"


augmentations = ["RandomGhosting", "RandomAnisotropy","RandomBiasField",
                 "RandomNoise", "RandomBlur",]


augmentation_name = augmentations[1]

all_files = []
for model in ["octree_nca", "m3d_nca", "unet"]:
    eval_file = pd.read_csv(f"{eval(model)}/eval/standard.csv", sep='\t')
    eval_file["severity"] = 0
    eval_file["model"] = model
    all_files.append(eval_file)
    for severity in range(1, 6):
        eval_file = pd.read_csv(f"{eval(model)}/eval/{augmentation_name}_{severity}.csv", sep='\t')
        eval_file["severity"] = severity
        eval_file["model"] = model
        all_files.append(eval_file)

eval_file = pd.concat(all_files)

sns.lineplot(data=eval_file, x="severity", y="DiceScore/0", hue="model", errorbar=None)
plt.title(augmentation_name)

plt.show()
