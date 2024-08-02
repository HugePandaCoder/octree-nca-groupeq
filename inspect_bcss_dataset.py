import matplotlib.pyplot as plt
import zarr, os, tqdm
import numpy as np
from PIL import Image


img_path=r"/local/scratch/BCSS/BCSS_TIF/images"
label_path= r"/local/scratch/BCSS/BCSS_TIF/masks"

files = os.listdir(img_path)

dict = {}
file = files[4]
for file in tqdm.tqdm(files):
    #image = Image.open(os.path.join(img_path, file))
    #image = np.array(image)
    label = Image.open(os.path.join(label_path, file))
    label = np.array(label)
    dict[file] = np.unique(label, return_counts=True)

for file in files:
    print(file, 1 in dict[file][0], 2 in dict[file][0], 3 in dict[file][0], 4 in dict[file][0])
#print(dict)
exit()
print(file)
print(image.shape)
print(label.shape)

#for c in range(mmapped_label.shape[-1]):
#    mmapped_label[...,c] *= c+1

#mmapped_label = np.max(mmapped_label, axis=-1)

print(np.any(label == 0))


plt.imshow(image)
plt.matshow(label, alpha=0.5, cmap='tab20')
plt.colorbar()

plt.show()