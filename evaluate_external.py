
import einops
import os, cv2
from src.losses.LossFunctions import DiceLoss
import numpy as np
import torch, tqdm
import matplotlib.pyplot as plt 

eval_path = "/local/scratch/clmn1/out_results_deeplab"
data_path = "/local/scratch/clmn1/data/cholecseg8k_preprocessed_2"
loss_f = DiceLoss(useSigmoid=True)
loss_log = {}
for c in range(5):
    loss_log[c] = {}


for patient in os.listdir(eval_path):
    print(patient)
    for video in tqdm.tqdm(os.listdir(os.path.join(data_path, patient))):
        gt = np.load(os.path.join(data_path, patient, video, "segmentation.npy"))#HWDC
        pred = np.load(os.path.join(eval_path, patient, video + ".npy")) #DHW
        
        pred = einops.rearrange(pred, 'd h w -> h w d')

        gt = einops.rearrange(gt, 'h w d c -> h w (d c)')
        gt = cv2.resize(gt, (424, 240), interpolation=cv2.INTER_NEAREST)
        gt = einops.rearrange(gt, 'h w (d c) -> h w d c', d=80)

        pred, gt = torch.from_numpy(pred), torch.from_numpy(gt)

        assert pred.shape == gt.shape[:3], f"{pred.shape} != {gt.shape[:3]}"

        if True:
            new_pred = torch.zeros_like(gt)
            for c in range(5):
                new_pred[:,:,:,c] = (pred == c+1).float()
        else:
            new_pred = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 6))
            new_pred[:,:,:,pred] = 1
            new_pred = new_pred[:,:,:,1:] # remove background lbl
        pred = new_pred


        #pred, gt = torch.from_numpy(pred), torch.from_numpy(gt)

        plt.imshow(pred[...,40, 0], cmap="gray")
        plt.imshow(gt[...,40, 0], alpha=0.5,)
        plt.colorbar()
        plt.show()
        exit()

        for m in range(gt.shape[-1]):
            if 1 in gt[..., m]:
                #class is present
                loss = 1- loss_f(gt[..., m], pred[..., m])
                loss_log[m][os.path.join(eval_path, patient, video + ".npy")] = loss.item()
            else:
                if 1 in pred[..., m]:
                    loss_log[m][os.path.join(eval_path, patient, video + ".npy")] = 0
                else:
                    continue

for key in loss_log:
    print(key, np.mean(list(loss_log[key].values())), np.std(list(loss_log[key].values())))

