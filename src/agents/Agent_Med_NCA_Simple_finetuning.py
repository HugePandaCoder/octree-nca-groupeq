import torch
import numpy as np
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import random
import torchio as tio
from matplotlib import pyplot as plt
from src.models.Model_Preprocess import PreprocessNCA
import torch.optim as optim
from itertools import chain
import torch.nn.functional as F

class Agent_Med_NCA_finetuning(MedNCAAgent):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """

    def initialize(self):
        # create test  model
        super().initialize()
        self.preprocess_model = PreprocessNCA(channel_n=16, fire_rate=0.3, device=self.device, hidden_size=256).to(self.device)
        self.preprocess_model2 = PreprocessNCA(channel_n=16, fire_rate=0.3, device=self.device, hidden_size=256).to(self.device)
        self.optimizer_test = optim.Adam(chain(self.preprocess_model.parameters(), self.preprocess_model2.parameters()), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler_test = optim.lr_scheduler.ExponentialLR(self.optimizer_test, self.exp.get_from_config('lr_gamma'))
        

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        

        if self.model.training:
            inputs, targets, inputs_loc = self.model(inputs, targets, return_channels=True, preprocess_model = (self.preprocess_model, self.preprocess_model2))
            return inputs, targets, inputs_loc
        else:
            inputs, targets = self.model(inputs, targets, return_channels=False, preprocess_model = (self.preprocess_model, self.preprocess_model2))
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
        outputs, targets, inputs_loc = self.get_outputs(data, return_channels=False)

        #plt.imshow(targets[0, :, :, 0].detach().cpu().numpy())
        #plt.show()

        #plt.imshow(((inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy()+1)/2)
        #plt.imshow((inputs_loc_3_ori[0, :, :, 0:1].detach().cpu().numpy()+1)/2)
        #plt.show()
        if False:
            self.exp.write_img('before',
                                    inputs_loc[1][0, :, :, 0:1].detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            self.exp.write_img('difference',
                                    torch.abs(inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            self.exp.write_img('after',
                                    inputs_loc[0][0, :, :, 0:1].detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
        else:
            difference = inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]
            difference = (difference - difference.min()) / (difference.max() - difference.min())
            cat_img =  torch.cat((inputs_loc[1][0, :, :, 0:1], difference, inputs_loc[0][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
            self.exp.write_img('preprocessing_main_level',
                                    cat_img,
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            difference = inputs_loc[3][0, :, :, 0:1] - inputs_loc[2][0, :, :, 0:1]
            difference = (difference - difference.min()) / (difference.max() - difference.min())
            cat_img =  torch.cat((inputs_loc[3][0, :, :, 0:1], difference, inputs_loc[2][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
            self.exp.write_img('preprocessing_patch_level',
                                    cat_img,
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
        #random.seed(rnd)
        #outputs2, targets, inputs_loc_2 = self.get_outputs(data, return_channels=True)

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

        #print("INPUTS LOC ", inputs_loc[0].shape)
        def toFourier(tensor):
            tensor = tensor.transpose(1, 3)
            tensor_loc = torch.fft.fft2(tensor, s = (12, 12), norm="forward")
            tensor = torch.fft.ifft2(tensor_loc, norm="forward", s=(tensor.shape[2], tensor.shape[3]))
            #x_start, x_end = tensor.shape[2]//2 -8, tensor.shape[2]//2 + 8
            #y_start, y_end = tensor.shape[3]//2 -8, tensor.shape[3]//2 + 8
            #tensor = torch.fft.fftshift(tensor)#[..., x_start:x_end, y_start:y_end]
            return tensor.real

        # >>>>> Loss MSE-variance between outputs 
        max_val = torch.max(torch.abs(inputs_loc[6][..., self.input_channels:]))
        loss = mse(inputs_loc[6][..., self.input_channels:] / max_val, inputs_loc[7][..., self.input_channels:] / max_val)
        
        # >>>>>Loss MSE-variance between mid layers
        max_val = torch.max(torch.abs(inputs_loc[4][..., self.input_channels:]))
        loss2 = mse(inputs_loc[4][..., self.input_channels:] / max_val, inputs_loc[5][..., self.input_channels:] / max_val)
            #l1(torch.log(torch.clamp(inputs_loc[2][..., self.input_channels:]*-1, 1e-10)), torch.log(torch.clamp(inputs_loc[3][..., self.input_channels:]*-1, 1e-10))))
        
        #loss2 = l1(inputs_loc[4][..., self.input_channels:], inputs_loc[5][..., self.input_channels:])
        
        # >>>>> Loss L1 between fourier
        loss3 = (mse(toFourier(inputs_loc[0][..., 0:self.input_channels]), toFourier(inputs_loc[1][..., 0:self.input_channels])) + \
            mse(toFourier(inputs_loc[2][..., 0:self.input_channels]), toFourier(inputs_loc[3][..., 0:self.input_channels])))
        
        # >>>>> Loss mse between fourier
        loss4 = (mse(apply_gaussian_blur(inputs_loc[0][..., 0:self.input_channels]), apply_gaussian_blur(inputs_loc[1][..., 0:self.input_channels])) + \
            mse(apply_gaussian_blur(inputs_loc[2][..., 0:self.input_channels]), apply_gaussian_blur(inputs_loc[3][..., 0:self.input_channels])))
        loss5 = (mse(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]) + \
                    mse(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels]))
                
        
        loss_ret = {}
        loss = ((loss + loss2) + loss4)*200 + loss5#loss3 #loss4 + 
        print(loss.item())
        loss_ret[0] = loss.item()
        #loss_ret[1] = loss2.item()
        #loss_ret[2] = loss3.item()

        

        weight_sum = 0
        for param in self.preprocess_model.parameters():
            weight_sum += param.data.sum()

        print("PARAM SUM", weight_sum)

        if loss != 0:
            loss.backward()

            self.optimizer_test.step()
            self.scheduler_test.step()
        return loss_ret
    

def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel with PyTorch."""
    coords = torch.arange(size).float()
    coords -= size // 2

    g = coords ** 2
    g = (-g / (2 * sigma ** 2)).exp()

    g /= g.sum()
    gaussian_kernel = torch.outer(g, g)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]

    return gaussian_kernel

def apply_gaussian_blur(image, kernel_size=13, sigma=2):
    """Applies Gaussian blur to an image using convolution."""
    channels = image.shape[1]
    gaussian_kernel_weights = gaussian_kernel(kernel_size, sigma).to(image.device)
    gaussian_kernel_weights = gaussian_kernel_weights.expand(channels, 1, kernel_size, kernel_size)
    
    # Ensuring the image has the channel in the second dimension, expected shape: [N, C, H, W]
    blurred_image = F.conv2d(image, gaussian_kernel_weights, padding=kernel_size//2, groups=channels)
    return blurred_image