import cv2
import os
from tqdm import tqdm

folder = r"/home/jkalkhof_locale/Documents/Data/img_align_celeba/"
out = r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/"



for filename in tqdm(os.listdir(folder)):
    path = os.path.join(folder, filename)
    out_path = os.path.join(out, filename)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(out_path, img)