import os, PIL, tqdm
from PIL import Image
import numpy as np
import zarr, openslide
PIL.Image.MAX_IMAGE_PIXELS = None

IMG_PATH = "/local/scratch/AGGC/AGGC2022_train/Subset1_Train_image/"
LABEL_PATH = "/local/scratch/AGGC/AGGC2022_train/Subset1_Train_annotations/"

IMG_OUT_PATH = "/local/scratch/clmn1/data/AGGC4/AGGC2022_train/Subset1_Train_image/"
LABEL_OUT_PATH = "/local/scratch/clmn1/data/AGGC4/AGGC2022_train/Subset1_Train_annotations/"

os.makedirs(LABEL_OUT_PATH, exist_ok=True)
os.makedirs(IMG_OUT_PATH, exist_ok=True)

num_for_loop_items = 0
for dir in tqdm.tqdm(os.listdir(LABEL_PATH)):
    num_for_loop_items += 1
    for file in os.listdir(os.path.join(LABEL_PATH, dir)):
        num_for_loop_items += 1


bar = tqdm.tqdm(range(num_for_loop_items))
bar_iter = iter(bar)
for dir in os.listdir(LABEL_PATH):
    bar.set_description(f"{dir}")
    next(bar_iter)
    os.makedirs(os.path.join(LABEL_OUT_PATH, dir), exist_ok=True)
    image = Image.open(os.path.join(IMG_PATH, dir + ".tiff"))
    image = np.array(image)
    zarr.save(os.path.join(IMG_OUT_PATH, dir + ".zarr"), image)


    for file in os.listdir(os.path.join(LABEL_PATH, dir)):
        bar.set_description(f"{dir} {file}")
        next(bar_iter)
        lbl = Image.open(os.path.join(LABEL_PATH, dir, file))
        lbl = np.array(lbl)


        assert lbl.shape == image.shape[:2], f"{lbl.shape} != {image.shape[:2]}"
        zarr.save(os.path.join(LABEL_OUT_PATH, dir, file[:-4] + ".zarr"), lbl)