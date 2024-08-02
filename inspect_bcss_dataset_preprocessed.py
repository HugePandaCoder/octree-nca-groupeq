import matplotlib.pyplot as plt
import zarr, os
import numpy as np

img_path=r"/local/scratch/clmn1/data/BCSS/BCSS_TIF/images/"
label_path= r"/local/scratch/clmn1/data/BCSS2/BCSS_TIF/masks/"

files = os.listdir(img_path)


file = files[4]
mmapped_image = zarr.open(os.path.join(img_path, file))
mmapped_label = zarr.open(os.path.join(label_path, file))

print(file)
print(mmapped_image.shape)
print(mmapped_label.shape)

#for c in range(mmapped_label.shape[-1]):
#    mmapped_label[...,c] *= c+1

#mmapped_label = np.max(mmapped_label, axis=-1)

plt.imshow(mmapped_image)
plt.imshow(mmapped_label[:,:, 2], alpha=0.5, cmap='tab20')

plt.colorbar()
plt.show()