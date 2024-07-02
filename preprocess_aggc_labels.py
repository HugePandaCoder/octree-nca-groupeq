import os, PIL, tqdm
from PIL import Image
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = None

INPUT_PATH = "/local/scratch/AGGC/AGGC2022_train/Subset1_Train_annotations/"
OUTPUT_PATH = "/local/scratch/clmn1/data/AGGC/AGGC2022_train/Subset1_Train_annotations/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

for dir in tqdm.tqdm(os.listdir(INPUT_PATH)):
    os.makedirs(os.path.join(OUTPUT_PATH, dir), exist_ok=True)
    for file in os.listdir(os.path.join(INPUT_PATH, dir)):
        img = Image.open(os.path.join(INPUT_PATH, dir, file))
        img = np.array(img)
        np.save(os.path.join(OUTPUT_PATH, dir, file[:-4] + ".npy"), img)