import os, PIL, tqdm
from PIL import Image
import numpy as np
import zarr, openslide
PIL.Image.MAX_IMAGE_PIXELS = None

IMG_PATH = "/local/scratch/AGGC/AGGC2022_train/Subset1_Train_image/"
INPUT_PATH = "/local/scratch/AGGC/AGGC2022_train/Subset1_Train_annotations/"
OUTPUT_PATH = "/local/scratch/clmn1/data/AGGC3/AGGC2022_train/Subset1_Train_annotations/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

for dir in tqdm.tqdm(os.listdir(INPUT_PATH)):
    os.makedirs(os.path.join(OUTPUT_PATH, dir), exist_ok=True)
    for file in os.listdir(os.path.join(INPUT_PATH, dir)):
        openslide_shape = openslide.OpenSlide(os.path.join(IMG_PATH, dir + ".tiff")).level_dimensions[0]

        lbl = Image.open(os.path.join(INPUT_PATH, dir, file))
        lbl = np.array(lbl)

        lbl = np.swapaxes(lbl, 0, 1)


        assert lbl.shape == openslide_shape, f"{lbl.shape} != {openslide_shape}"
        zarr.save(os.path.join(OUTPUT_PATH, dir, file[:-4] + ".zarr"), lbl)