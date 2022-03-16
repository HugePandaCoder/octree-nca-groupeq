import nibabel as nib
import numpy as np
import os

class BaseAgent():
    def __init__(self):
        return

    def saveNiiGz(self, output, label, patient_id, path):
        output = np.round(output.cpu().detach().numpy())
        output[output < 0] = 0
        output[output > 0] = 1
        nib_image = nib.Nifti1Image(output, np.eye(4))
        nib_label = nib.Nifti1Image(label.cpu().detach().numpy(), np.eye(4))
        nib.save(nib_image, os.path.join(path, patient_id + "_image.nii.gz"))  
        nib.save(nib_label, os.path.join(path, patient_id + "_label.nii.gz"))  
        return
