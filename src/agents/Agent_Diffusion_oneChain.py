import numpy as np
from src.agents.Agent_NCA import Agent_NCA
from src.agents.Agent_Diffusion import Agent_Diffusion
import torch
import torchvision
import random
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import cv2
import os
import time
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

class Agent_Diffusion_Chain(Agent_Diffusion):
    def get_outputs(self, data, full_img=False, t=0, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        t = torch.tensor(t/self.timesteps).to(self.device) # TODO: torch.tensor((t+1)/self.timesteps).to(self.device)
        #print("T", t)
        id, inputs, targets = data['id'], data['image'], data['label']
        inputs = inputs.to(self.device).to(torch.float)
        if self.exp.model_state == "train":
            #print("TTTTTTT<-------------------", t.shape)
            t = t.repeat(inputs.shape[0])
        
        noise_chain = []
        noise_pred_chain = []

        outputs = None
        # Whole denoising path

        timesteps = 0
        if self.model[0].training:
            timesteps = self.exp.get_from_config('timesteps_train')
            #random_start = random.randint(0, self.timesteps - timesteps)

            t = torch.randint(timesteps, self.timesteps, (data['image'].shape[0],), device=self.exp.get_from_config(tag="device")).long()
            
        else:
            t = torch.zeros((data['image'].shape[0],), device=self.exp.get_from_config(tag="device")).long()
            timesteps = self.timesteps




        for i in reversed(range(timesteps)):
            t_loc = t#+i
            # Same timestep all
            #ts = ts.repeat(inputs.shape[0])

            # Put Noise on image

            with torch.no_grad():
                data_noisy = self.prepare_data(data, t_loc) # <-- QSAMPLE
            ts_div = t_loc / self.timesteps


            image_noisy = data_noisy['image']
            noise = data_noisy['noise']

            if outputs is not None:
                image_noisy[..., self.input_channels:] = outputs[..., self.input_channels:] 

            outputs = self.model[0](image_noisy, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=ts_div, epoch=self.exp.currentStep)


            # now psample
            if self.model[0].training is False:
                data['image'] = self.p_sample(outputs[..., self.input_channels:self.input_channels+self.output_channels], # noise
                                        image_noisy[..., :self.input_channels], 
                                        t_loc,
                                        t_loc)

            # Store inbetween steps
            noise_chain.append(noise)
            noise_pred_chain.append(outputs[..., self.input_channels:self.input_channels+self.output_channels])
        

        outputs = outputs[..., self.input_channels:self.input_channels+self.output_channels]

        return outputs, noise_chain, noise_pred_chain, noise
    
    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        #if isinstance(self.model, list):
        #    rang = int((self.timesteps / len(self.model)) * np.random.randint(0, len(self.model)))
        #    t = torch.randint(0, int(self.timesteps / len(self.model)), (data['image'].shape[0],), device=self.exp.get_from_config(tag="device")).long()
        #    t = torch.add(t, rang)
        #    #print(t)
        #else:
        #    t = torch.randint(0, self.timesteps, (data['image'].shape[0],), device=self.exp.get_from_config(tag="device")).long()

        scaler = GradScaler()

        real_img = data['image'].to(self.device).to(torch.float)

        # Rescale img factor x
        #data = self.rescale_image(data)
        
        #data = self.prepare_data(data, t) #data, noise, label = self.prepare_data
        id, img, _ = data['id'], data['image'], data['label']
        #with autocast():
        outputs, noise_chain, noise_pred_chain, noise = self.get_outputs(data, t=0)

        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                self.optimizer[m].zero_grad()
        else:
            self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}

        # Predict image but go denoising path
        for e in range(len(noise_chain)):
            #loss += F.mse_loss( noise_pred_chain[e])

            #plt.imshow((noise_pred_chain[e][0, :, :, 0:3].detach().cpu().numpy()+1)/2)
            #plt.show()
            loss_mse = F.mse_loss(outputs, noise, reduction='none')
            loss_mse = loss_mse.mean(dim=1)
            loss_l1 = F.l1_loss(outputs, noise, reduction='none')
            loss_l1 = loss_l1.mean(dim=1)

            loss = loss + loss_mse.mean() + loss_l1.mean()


        #loss = loss_mse.mean() + loss_l1.mean() #+ (loss_mse_fourier_magnitude.mean()*0.1 + loss_mse_fourier_phase.mean()) * 0.001

        loss_ret[0] = loss

        #scaler.scale(loss).backward()
        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                self.optimizer[m].step()
                #scaler.step(self.optimizer[m])#.step()
        else:
            self.optimizer.step()
            #scaler.step(self.optimizer)#.step()
        if isinstance(self.scheduler, list): 
            for m in range(len(self.scheduler)):
                self.scheduler[m].step()
        else:
            self.scheduler.step()
        #scaler.update()
        return loss_ret


    # Prepare denoising timeline
    def prepare_data(self, data, t, label=None, eval=False):
        r"""
        preprocessing of data
        :param data: images
        :param t: current time steps
        :param batch_size:
        :param label:
        :return: corrupt images, associated noise
        """
        id, img, _ = data['id'], data['image'], data['label']
        img = img.to(self.device)

        if self.model[0].training is False:
            noise = img
            img_noisy = img
        else:
            noise, img_noisy = self.getNoiseLike(img, noisy=True, t=t)


        
        img_noisy = self.make_seed(img_noisy)
        if self.model[0].training is False:
            img_noisy, noise = self.repeatBatch(img_noisy, noise, self.exp.get_from_config('batch_duplication'))
        data_noisy = (id, img_noisy, img_noisy)

        data = {'id': id, 'image': img_noisy, 'label': img_noisy, 'noise': noise}

        return data
    
    @torch.no_grad()
    def test(self, tag='0', samples=1, extra=False, normal=True, **kwargs):

        # Generate sample
        size = self.exp.get_from_config('input_size')
        
        dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        #self.test_fid()
        #noise = torch.randn_like(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels')))).to(self.device)

        #self.timesteps = 300



        if normal:
            for s in range(samples):
                if True:
                    noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))

                    data = {'id': 0,'image': noise,'label': noise}
                    print("NOISE", data['image'].shape)
                    output, *_ = self.get_outputs(data, t = 0)

                    print(output.shape)
                    self.exp.write_img(tag, (output[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}