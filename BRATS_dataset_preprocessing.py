import os
import nibabel as nib
import numpy as np

labels_directory = r"/home/jkalkhof_locale/Documents/Data/Task001_BrainTumour/labelsTr/"
images_directory = r"/home/jkalkhof_locale/Documents/Data/Task001_BrainTumour/imagesTr/"
out_dir = r"/home/jkalkhof_locale/Documents/Data/Task001_BrainTumour_fixed/imagesTr/"

for file in os.listdir(labels_directory):
    name_out = os.path.basename(file)
    name = name_out[:-7] 

    img_base = nib.load(os.path.join(images_directory, name + "_0000.nii.gz"))
    img0 = img_base.get_fdata()
    img1 = nib.load(os.path.join(images_directory, name + "_0001.nii.gz")).get_fdata()
    img2 = nib.load(os.path.join(images_directory, name + "_0002.nii.gz")).get_fdata()
    img3 = nib.load(os.path.join(images_directory, name + "_0003.nii.gz")).get_fdata()

    img_out = np.stack((img0, img1, img2, img3), axis=-1)

    nib_img = nib.Nifti1Image(img_out, img_base.affine, img_base.header)

    nib.save(nib_img, os.path.join(out_dir, name_out))






