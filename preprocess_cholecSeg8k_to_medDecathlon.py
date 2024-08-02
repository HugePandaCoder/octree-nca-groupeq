import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import os
import pathlib, cv2, einops, math
from nnunet.dataset_conversion.utils import generate_dataset_json


def reshape_batch(instack, size: tuple, is_label:bool = False,) -> np.ndarray:
    #https://stackoverflow.com/questions/65154879/using-opencv-resize-multiple-the-same-size-of-images-at-once-in-python
    N,H,W,C = instack.shape
    instack = instack.transpose((1,2,3,0)).reshape((H,W,C*N))

    outstacks = []
    for i in range(math.ceil(instack.shape[-1] / 500)):
        if is_label:
            outstack = cv2.resize(instack[..., i*500:(i+1)*500], (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            outstack = cv2.resize(instack[..., i*500:(i+1)*500], (size[1], size[0]))
        outstacks.append(outstack)

    outstack = np.concatenate(outstacks, axis=-1)
    return outstack.reshape((size[0], size[1], C, N)).transpose((3,0,1,2))



input_folder = "/local/scratch/clmn1/data"
#output_folder = "/local/scratch/clmn1/data/medDecathlon"
output_folder = "/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/nnUNet_raw_data"
task_name = "Task500_cholecseg8k"


color_class_mapping={
                            #(127, 127, 127): 0,
                            (140, 140, 210): 1,     #Abdominal wall        
                            (114, 114, 255): 2,     #Liver                 
                            #(156, 70, 231): 3,     #Gastrointestinal tract
                            (75, 183, 186): 4,      #Fat                   
                            (0, 255, 170): 5,       #Grasper               
                            #(0, 85, 255): 6,       #Connective tissue     
                            #(0, 0, 255): 7,        #Blood                 
                            #(0, 255, 255): 8,      #Cystic duct           
                            #(184, 255, 169): 9,    #L-hook electrocautery   
                            (165, 160, 255): 10,    #Gallbladder             
                            #(128, 50, 0): 11,      #Heptatic vein         
                            #(0, 74, 111): 12       #Liver ligament        
                            }

color_classes = [k for k in color_class_mapping.keys()]

label_names = {0: "background",
               1: "Abdominal wall",
               2: "Liver",
               3: "Fat",
               4: "Grasper",
               5: "Gallbladder"}
               


os.makedirs(os.path.join(output_folder, task_name, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_folder, task_name, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(output_folder, task_name, "imagesTs"), exist_ok=True)

files = []


for patient in os.listdir(os.path.join(input_folder, "cholecseg8k")):
    print(patient)
    if not os.path.isdir(os.path.join(input_folder, "cholecseg8k", patient)):
        continue
    for f in os.listdir(os.path.join(input_folder, "cholecseg8k", patient)):
        first_frame = int(f[len("videoXX_"):])
        path = os.path.join(input_folder, "cholecseg8k", patient, f)

        imgs = []
        lbls = []
        for frame in range(first_frame, first_frame + 80):
            label = cv2.imread(os.path.join(path, f"frame_{frame}_endo_color_mask.png"))
            lbls.append(label)
            image = cv2.imread(os.path.join(path, f"frame_{frame}_endo.png"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.append(image)
        
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        

        imgs = imgs.astype(np.float32)
        imgs /= 255.0

        imgs = einops.rearrange(imgs, 'd h w c -> c h w d')
        lbls = einops.rearrange(lbls, 'd h w c -> h w d c')

        new_labels = np.zeros((lbls.shape[0], lbls.shape[1], lbls.shape[2], 1), dtype=np.uint8)


        for i, k in enumerate(color_classes):
            mask = np.all(lbls == k, axis=-1)
            new_labels[mask] = i+1

        imgs = einops.rearrange(imgs, 'c h w d -> d h w c')
        imgs = reshape_batch(imgs, (240, 432))

        new_labels = einops.rearrange(new_labels, 'h w d c -> d h w c')
        new_labels = reshape_batch(new_labels, (240, 432), is_label=True)

        print(imgs.shape)
        print(new_labels.shape)

        for channel in range(3):
            sitk.WriteImage(sitk.GetImageFromArray(imgs[..., channel]), os.path.join(output_folder, task_name, "imagesTr", f"{f}_000{channel}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(new_labels), os.path.join(output_folder, task_name, "labelsTr", f"{f}.nii.gz"))

        print(patient, f)
        


generate_dataset_json(os.path.join(output_folder, task_name, "dataset.json"),
                      os.path.join(output_folder, task_name, "imagesTr"),
                      os.path.join(output_folder, task_name, "imagesTs"),
                      ("R", "G", "B"), label_names, "Task500_cholecseg8k")
