import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
from src.scores import ScoreList
from src.agents.Agent import BaseAgent
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math 
import torch.utils.data
import torchio as tio

class Agent_MedSeg3D(BaseAgent):

    @torch.no_grad()
    def test(self, scores: ScoreList, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, 
             split='test', ood_augmentation: tio.Transform=None,
             output_name: str=None) -> dict:
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.exp.set_model_state('test')
        
        # Prepare arrays
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        patient_real_Img = None
        loss_log = {}
        if save_img == None:
            save_img = []#1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            assert data['image'].shape[0] == 1, "Batch size must be 1 for evaluation"

            if ood_augmentation != None:
                data['image'] = ood_augmentation(data['image'][0])
                data["image"] = data["image"][None]

            data_id, inputs, targets = data['id'], data['image'], data['label']
            out = self.get_outputs(data, full_img=True, tag="0")
            outputs = out["pred"]

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                print("DATA_ID", data_id)
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None


            # Run inference 10 times to create a pseudo ensemble
            if pseudo_ensemble: # 5 + 5 times
                outputs2 = self.get_outputs(data, full_img=True, tag="1")["pred"]
                outputs3 = self.get_outputs(data, full_img=True, tag="2")["pred"]
                outputs4 = self.get_outputs(data, full_img=True, tag="3")["pred"]
                outputs5 = self.get_outputs(data, full_img=True, tag="4")["pred"]
                if True: 
                    outputs6 = self.get_outputs(data, full_img=True, tag="5")["pred"]
                    outputs7 = self.get_outputs(data, full_img=True, tag="6")["pred"]
                    outputs8 = self.get_outputs(data, full_img=True, tag="7")["pred"]
                    outputs9 = self.get_outputs(data, full_img=True, tag="8")["pred"]
                    outputs10 = self.get_outputs(data, full_img=True, tag="9")["pred"]
                    stack = torch.stack([outputs, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10], dim=0)
                    
                    # Calculate median
                    outputs, _ = torch.median(stack, dim=0)
                    self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy() )

                else:
                    outputs, _ = torch.median(torch.stack([outputs, outputs2, outputs3, outputs4, outputs5], dim=0), dim=0)

            patient_3d_image = outputs.detach().cpu()
            patient_3d_label = targets.detach().cpu()
            patient_3d_real_Img = inputs.detach().cpu()
            patient_id = id
            #print(patient_id)

            s = scores(patient_3d_image, patient_3d_label)
            for key in s.keys():
                if key not in loss_log:
                    loss_log[key] = {}
                loss_log[key][patient_id] = s[key]

                print(",",loss_log[key][patient_id])
                # Add image to tensorboard
                if True: 
                    if len(patient_3d_label.shape) == 4:
                        patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                    #self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                    #convert_image(self.prepare_image_for_display(patient_3d_real_Img[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    #self.prepare_image_for_display(patient_3d_image[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    #self.prepare_image_for_display(patient_3d_label[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    #encode_image=False), self.exp.currentStep)

                    # REFACTOR: Save predictions
                    if False:
                        label_out = torch.sigmoid(patient_3d_image[0, ...])
                        nib_save = nib.Nifti1Image(label_out  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                        nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + ".nii.gz"))

                        nib_save = nib.Nifti1Image(torch.sigmoid(patient_3d_real_Img[0, ...])  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                        nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_real.nii.gz"))

                        nib_save = nib.Nifti1Image(patient_3d_label[0, ...]  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                        nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_ground.nii.gz"))

            #print(patient_3d_real_Img.shape, patient_3d_image.shape, patient_3d_label.shape)
            if ood_augmentation is None:
                self.exp.write_img(str(tag) + str(patient_id),
                                merge_img_label_gt_simplified(patient_3d_real_Img, patient_3d_image, patient_3d_label, rgb=dataset.is_rgb),
                                self.exp.currentStep)
        
        ood_label = ""
        if ood_augmentation != None:
            ood_label = str(ood_augmentation)

        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                print(f"Average Dice Loss {ood_label} 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                print(f"Standard Deviation {ood_label} 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

        self.exp.set_model_state('train')


        return loss_log