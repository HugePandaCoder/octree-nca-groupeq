import os 
import nibabel as nib
from src.losses.LossFunctions import DiceLoss
import torch

location = "M:\\MasterThesis\\Git\\UKBB\\demo_image"

dice = DiceLoss(useSigmoid = False)

loss_list = {}

for folder in os.listdir(location):
    for file_name in os.listdir(os.path.join(location, folder)):
        if ".nii.gz" in file_name and "label" in file_name:# and "prediction" not in file_name[-10:]:
            label_name = os.path.join(location, folder, file_name)
            prediction_name = label_name.replace("label", "prediction")
            
            label_data = nib.load(label_name).get_fdata()
            prediction_data = nib.load(prediction_name).get_fdata()
            prediction_data = prediction_data[:,:,:,0]

            label_data = torch.tensor(label_data)
            prediction_data = torch.tensor(prediction_data)

            #print("________________________")

            name = file_name[:-13]

            loss_list[name] = {}

            for a in [1, 2, 3]:
                label_data2 = label_data.clone()
                prediction_data2 = prediction_data.clone()

                label_data2[label_data2 != a] = 0
                label_data2[label_data2 != 0] = 1

                prediction_data2[prediction_data2 != a] = 0
                prediction_data2[prediction_data2 != 0] = 1
                
                value = float(1 - dice(prediction_data2, label_data2, smooth=0))
                loss_list[name][a] = value
                #print(value)
                #loss_list[a-1].append(value) #loss_list[a-1] = 
                #print(value)

            #loss_list.append(float(1 - dice(prediction_data, label_data, smooth=0)))
            #print(loss_list[-1])

print("AVERAGE")
#for a in [1, 2, 3]:
print(loss_list)

for patient in loss_list:
    print(str(patient) + ", " + str(loss_list[patient][1]) + ", " + str(loss_list[patient][2]) + ", " + str(loss_list[patient][3]))
    #print(sum(loss_list[a-1]) / len(loss_list[a-1]))
#print(sum(loss_list) / len(loss_list))