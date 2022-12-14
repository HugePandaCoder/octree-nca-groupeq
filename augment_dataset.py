import os
import nibabel as nib
import numpy as np 
import torchio

dataset_path = r"/home/jkalkhof_locale/Documents/Data/Prostate_Augmented/Prostate_Full_Combined_PaperOut/data"
out_path = r"/home/jkalkhof_locale/Documents/Data/Prostate_Augmented/Prostate_Full_Combined_PaperOut/BiasField"

#dataset_path = r"/home/jkalkhof_locale/Documents/GitHub/nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task117_prostate_test_MNMs/labelsTr/"
#out_path = r"/home/jkalkhof_locale/Documents/Data/Prostate_Augmented/Prostate_Full_Combined_Test_Translation4375/labelsTr/"

severity = 6

for file in os.listdir(dataset_path):
    nib_img = nib.load(os.path.join(dataset_path, file))
    data = nib_img.get_fdata()
    #print(data.shape)

    if len(data.shape) == 4:
        data = data[...,0]

    # Augmentation
    if False:
        transform = torchio.transforms.RandomGhosting(num_ghosts=(severity,severity), intensity=severity/4, axes=0, restore=0) #num_ghosts=2, intensity=2,
        data = np.expand_dims(data, axis=0)
        data = transform(data)
        data = np.squeeze(data)

    if False:
        transform = torchio.transforms.RandomAnisotropy(axes=1, downsampling=severity) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
        data = np.expand_dims(data, axis=0)
        data = transform(data)
        data = np.squeeze(data)

    if True:
        transform = torchio.transforms.RandomBiasField(coefficients = 0.1 * severity, order = 3) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
        data = np.expand_dims(data, axis=0)
        for z in range(data.shape[3]):
            slice = data[...,z]
            slice = slice[..., np.newaxis] 
            slice = transform(slice)
            slice = slice[0,:,:,0] 
            data[0,:,:,z] = slice
        data = np.squeeze(data)

    if False:
        print(data.shape)
        size = data.shape[0]
        print(size) 
        target = int(((severity/100)/2) * size)
        #data[0:target, :, :] = 0
        #data[size-target:size,:,  :] = 0  
        data[:, 0:target, :] = 0
        data[:, size-target:size, :] = 0  

    if False:
        #Real CROP
        print(data.shape)
        size = data.shape[0]
        print(size) 
        target = int(((severity/100)/2) * size)
        #data[0:target, :, :] = 0
        #data[size-target:size,:,  :] = 0  
        data = data[:, target:size-target, :] 
        #[:, 0:target, :] = 0
        #data[:, size-target:size, :] = 0 

    if False:
        size = data.shape[0]
        target = int((severity/100) * size)
        data = np.roll(data, target, axis=1)
        data[:,0:target, :] = 0 


    nib_save = nib.Nifti1Image(data, np.eye(4), nib_img.header)
    nib.save(nib_save, os.path.join(out_path, file))