import nibabel as nib
import numpy as np

label_name = r"D:\CMRxMotion_Training_Dataset\ukbb_cardiac_predictions\full\P004-1\P004-1-ED-prediction.nii.gz"
out = r"C:\Users\John\Desktop\test_label.nii.gz"

nib_img = nib.load(label_name)
mask = nib_img.get_fdata()
mask = np.roll(mask, -50, axis=0)
mask = mask[0:400, : , :]


nib_label = nib.Nifti1Image(mask, nib_img.affine)

nib.save(nib_label, out)

