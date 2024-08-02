import os, PIL, tqdm
from PIL import Image
import numpy as np
import zarr, openslide

#https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv
#https://academic.oup.com/bioinformatics/article/35/18/3461/5307750#supplementary-data


# class      count
# 0     889,839,209
# 1   1,632,574,020
# 2   1,422,973,342
# 3     372,961,701
# 4     256,565,874
# 5       2,152,072
# 6       9,559,670
# 7     110,979,770
# 8       2,965,951
# 9      43,241,823
# 10     75,007,895
# 11     11,790,134
# 12        138,927
# 13      3,978,889
# 14        408,630
# 15      3,875,212
# 16         50,590
# 17      2,292,470
# 18     22,841,931
# 19      1,128,660
# 20      2,836,030




IMG_PATH = "/local/scratch/BCSS/BCSS_TIF/images"
LABEL_PATH = "/local/scratch/BCSS/BCSS_TIF/masks"

IMG_OUT_PATH = "/local/scratch/clmn1/data/BCSS/BCSS_TIF/images/"
LABEL_OUT_PATH = "/local/scratch/clmn1/data/BCSS2/BCSS_TIF/masks/"

os.makedirs(LABEL_OUT_PATH, exist_ok=True)
os.makedirs(IMG_OUT_PATH, exist_ok=True)


all_labels = {}

for img_name in os.listdir(LABEL_PATH):
    print("Processing", img_name)
    #image = Image.open(os.path.join(IMG_PATH, img_name))
    #image = np.array(image)
    #zarr.save(os.path.join(IMG_OUT_PATH, img_name[:-len(".tiff")] + ".zarr"), image)

    label = Image.open(os.path.join(LABEL_PATH, img_name))
    label = np.array(label)
    #assert label.shape == image.shape[:2],  f"{label.shape} != {image.shape[:2]}"

    label_copy = np.zeros((*label.shape, 4))

    for i in range(1, 4+1):
        label_copy[label == i] = 1


    #print("")

    zarr.save(os.path.join(LABEL_OUT_PATH, img_name[:-len(".tiff")] + ".zarr"), label_copy)
