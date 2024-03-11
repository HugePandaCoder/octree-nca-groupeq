#%%
import numpy as np
from PIL import Image
import os
from src.losses.LossFunctions import DiceLoss
import nibabel as nib
import torch
from matplotlib import pyplot as plt
import math 
def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss (1 - Dice Coefficient) between two binary images.
    
    Args:
    - y_true: Ground truth binary image (numpy array).
    - y_pred: Predicted binary image (numpy array).
    
    Returns:
    - Dice loss as a float.
    """
    dice = DiceLoss(useSigmoid=False)

    return 1 - dice(y_true, y_pred)

def load_nifti_image(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def load_matching_images(predictions_folder, ground_truth_folder):
    predictions = []
    ground_truths = []
    for filename in os.listdir(predictions_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            prediction_path = os.path.join(predictions_folder, filename)
            ground_truth_path = os.path.join(ground_truth_folder, filename)
            
            # Load the prediction and its corresponding ground truth
            pred_img = load_nifti_image(prediction_path)
            #print(prediction_path, ground_truth_path)
            gt_img = load_nifti_image(ground_truth_path[:-3])
            
            predictions.append(pred_img)
            ground_truths.append(gt_img)
    return predictions, ground_truths

def standard_deviation(loss_log: dict) -> float:
    r"""Calculate the standard deviation
        #Args
            loss_log: losses
    """
    mean = sum(loss_log)/len(loss_log)
    stdd = 0
    for e in loss_log:
        stdd = stdd + pow(e - mean, 2)
    stdd = stdd / len(loss_log)
    stdd = math.sqrt(stdd)
    return stdd

# Paths to the folders
comp = 33
nnUNetpath = '/home/jkalkhof_locale/Documents/MICCAI24_finetuning/nnUNet_results/2d'

localPath = os.path.join(nnUNetpath, 'Task010_ChestX8_50/nnUNetTrainerV2__nnUNetPlansv2.1/head_Task010_ChestX8_50/fold_0/')
if comp == 11:
    predictions_folder = os.path.join(localPath, 'Preds_Task011_MIMIC_50/')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels_test'
if comp == 12:
    predictions_folder = os.path.join(localPath, 'Preds_Task010_ChestX8_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/labels_test'
if comp == 13:
    predictions_folder = os.path.join(localPath, 'Preds_Task012_Padchest_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/labels_test'

localPath = os.path.join(nnUNetpath, 'Task011_MIMIC_50/nnUNetTrainerV2__nnUNetPlansv2.1/head_Task011_MIMIC_50/fold_0/')
if comp == 21:
    predictions_folder = os.path.join(localPath, 'Preds_Task011_MIMIC_50/')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels_test'
if comp == 22:
    predictions_folder = os.path.join(localPath, 'Preds_Task010_ChestX8_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/labels_test'
if comp == 23:
    predictions_folder = os.path.join(localPath, 'Preds_Task012_Padchest_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/labels_test'

localPath = os.path.join(nnUNetpath, 'Task012_Padchest_50/nnUNetTrainerV2__nnUNetPlansv2.1/head_Task012_Padchest_50/fold_0/')
if comp == 31:
    predictions_folder = os.path.join(localPath, 'Preds_Task011_MIMIC_50/')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels_test'
if comp == 32:
    predictions_folder = os.path.join(localPath, 'Preds_Task010_ChestX8_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/labels_test'
if comp == 33:
    predictions_folder = os.path.join(localPath, 'Preds_Task012_Padchest_50')
    ground_truth_folder = '/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/labels_test'

# Load images
predictions, ground_truths = load_matching_images(predictions_folder, ground_truth_folder)

# Calculate Dice Loss for each image
dice_losses = []
for pred, gt in zip(predictions, ground_truths):
    pred = torch.tensor(pred, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)

    gt = np.clip(gt[..., 0] + gt[..., 1], 0, 1)
    pred = pred[..., 0]

    imshow = torch.cat([pred, gt], dim=1)

    #plt.imshow(imshow.numpy())
    #plt.show()

    dice_losses.append(dice_loss(gt, pred))

# Calculate average Dice Loss
print(dice_losses)
average_dice_loss = np.mean(dice_losses)
std_dice_loss = standard_deviation(dice_losses)
print(f'Average Dice Loss: {average_dice_loss}')
print(f'Standard Deviation of Dice Loss: {std_dice_loss}')
# %%
