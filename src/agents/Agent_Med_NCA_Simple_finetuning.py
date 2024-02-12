import torch
import numpy as np
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import random
import torchio as tio
from matplotlib import pyplot as plt
from src.models.Model_Preprocess import PreprocessNCA
import torch.optim as optim

class Agent_Med_NCA_finetuning(MedNCAAgent):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """

    def initialize(self):
        # create test  model
        super().initialize()
        self.preprocess_model = PreprocessNCA(channel_n=16, fire_rate=0.5, device=self.device, hidden_size=128).to(self.device)
        self.optimizer_test = optim.Adam(self.preprocess_model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler_test = optim.lr_scheduler.ExponentialLR(self.optimizer_test, self.exp.get_from_config('lr_gamma'))
        

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        

        if self.model.training:
            inputs, targets, inputs_loc = self.model(inputs, targets, return_channels=True, preprocess_model = self.preprocess_model)
            return inputs, targets, inputs_loc
        else:
            inputs, targets = self.model(inputs, targets, return_channels=False, preprocess_model = self.preprocess_model)
            return inputs, targets

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        rnd = random.randint(0, 1000000000)
        random.seed(rnd)
        outputs, targets, inputs_loc = self.get_outputs(data, return_channels=True)

        #plt.imshow(targets[0, :, :, 0].detach().cpu().numpy())
        #plt.show()

        #plt.imshow(((inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy()+1)/2)
        #plt.imshow((inputs_loc_3_ori[0, :, :, 0:1].detach().cpu().numpy()+1)/2)
        #plt.show()
        self.exp.write_img('before',
                                inputs_loc[1][0, :, :, 0:1].detach().cpu().numpy(),
                                #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                self.exp.currentStep)
        self.exp.write_img('difference',
                                (inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy(),
                                #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                self.exp.currentStep)
        self.exp.write_img('after',
                                inputs_loc[0][0, :, :, 0:1].detach().cpu().numpy(),
                                #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                self.exp.currentStep)

        random.seed(rnd)
        outputs2, targets, inputs_loc_2 = self.get_outputs(data, return_channels=True)

        #plt.imshow(targets[0, :, :, 0].detach().cpu().numpy())
        #plt.show()

        self.optimizer_test.zero_grad()
        loss = 0
        #print(outputs.shape, targets.shape)
        #if len(outputs.shape) == 5:
        #    for m in range(targets.shape[-1]):
        #        loss_loc = loss_f(outputs[..., m], targets[...])
        #        loss = loss + loss_loc
        #        loss_ret[m] = loss_loc.item()
        #else:
        #    for m in range(targets.shape[-1]):
        #        if 1 in targets[..., m]:
        #            loss_loc = loss_f(outputs[..., m], targets[..., m])
        #            loss = loss + loss_loc
        #            loss_ret[m] = loss_loc.item()


        mse = torch.nn.MSELoss()
        l1 = torch.nn.L1Loss()
        #loss = mse(torch.sum(torch.sigmoid(outputs)), torch.sum(torch.sigmoid(outputs2)))
        #target = torch.sigmoid(outputs2)
        #target[target > 0.5] = 1
        #target[target < 0.5] = 0
        loss = l1(torch.log(torch.abs(outputs)), torch.log(torch.abs(outputs2))) + 10*(l1(inputs_loc[0], inputs_loc[1]) + l1(inputs_loc[2], inputs_loc[3]) + l1(inputs_loc_2[0], inputs_loc_2[1]) + l1(inputs_loc_2[2], inputs_loc_2[3]))
        print(loss.item())
        loss_ret = {}
        loss_ret[0] = loss.item()

        weight_sum = 0
        for param in self.preprocess_model.parameters():
            weight_sum += param.data.sum()

        print("PARAM SUM", weight_sum)

        if loss != 0:
            loss.backward()

            self.optimizer_test.step()
            self.scheduler_test.step()
        return loss_ret