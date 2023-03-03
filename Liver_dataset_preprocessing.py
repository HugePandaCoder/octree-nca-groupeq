import os
import nibabel as nib
import numpy as np
import torchio as tio

labels_directory = r"/home/jkalkhof_locale/Documents/Data/Task003_Liver/labelsTr/"
images_directory = r"/home/jkalkhof_locale/Documents/Data/Task003_Liver/imagesTr/"
out_dir = r"/home/jkalkhof_locale/Documents/Data/Task003_Liver_Scaled/"

transformImg = False

for file in os.listdir(labels_directory):
    name_out = os.path.basename(file)
    name = name_out[:-7] 

    if transformImg:
        img_base = nib.load(os.path.join(images_directory, file))# + "_0000.nii.gz"))
        transform = tio.Resample((2, 2, 12))
        img_base = transform(img_base)
        nib.save(img_base, os.path.join(out_dir, 'imagesTr', name_out))

    label_base = nib.load(os.path.join(labels_directory, file))
    transform_label = tio.Resample((2, 2, 12), image_interpolation='nearest')
    label_base = transform_label(label_base)
    nib.save(label_base, os.path.join(out_dir, 'labelsTr', name_out))




    #img0 = img_base.get_fdata()
    #img1 = nib.load(os.path.join(images_directory, name + "_0001.nii.gz")).get_fdata()
    #img2 = nib.load(os.path.join(images_directory, name + "_0002.nii.gz")).get_fdata()
    #img3 = nib.load(os.path.join(images_directory, name + "_0003.nii.gz")).get_fdata()

    #img_out = np.stack((img0, img1, img2, img3), axis=-1)

    
    

    
    

    #nib_img = nib.Nifti1Image(img_base, img_base.affine, img_base.header)

    
    






