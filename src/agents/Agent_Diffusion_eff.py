import numpy as np
from src.agents.Agent_NCA import Agent_NCA
from src.agents.Agent_Multi_NCA import Agent_Multi_NCA
import torch
import random
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import cv2
import os
import time

class Agent_Diffusion(Agent_Multi_NCA):
    def initialize(self, beta_schedule='linear'): #'linear'): cosine
        super().initialize()
        self.timesteps = self.exp.get_from_config('timesteps')
        self.beta_schedule = self.exp.get_from_config('schedule') #beta_schedule
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.betas, self.sqrt_recip_alphas, \
            self.posterior_variance = self.calc_schedule()
        self.averages = False

        self.gif = 0

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):

        if False:
            sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )

            noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        else:
            x_start = x_start.transpose(1,3)
            x_start = torch.fft.fft2(x_start, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
            x_start = torch.fft.fftshift(x_start, dim=(2,3))

            noise = noise.transpose(1,3)
            noise = torch.fft.fft2(noise, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
            noise = torch.fft.fftshift(noise, dim=(2,3))

            for b in range(x_start.shape[0]):
                patch = int(t[b])*2 +1
                half_x = x_start.shape[2]//2 - patch//2
                half_y = x_start.shape[3]//2 - patch//2

                #x_temp = x_start[b, :, half_x:half_x+patch, half_y:half_y+patch].clone()
                #x_start[b, :, :, :] = x_start[b, :, :, :] + noise[b, :, :, :] 
                #x_start[b, :, half_x:half_x+patch, half_y:half_y+patch] = x_temp

                noise[b, :, half_x:half_x+patch, half_y:half_y+patch] = 0

            noise = torch.fft.ifftshift(noise, dim=(2,3))
            noise = torch.fft.ifft2(noise, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"

            x_start = torch.fft.ifftshift(x_start, dim=(2,3))
            x_start = torch.fft.ifft2(x_start, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"
            
            x_start = noise + x_start
            x_start = x_start.transpose(1,3)
            
            #print(t[0])
            #plt.imshow((x_start.real[0, :, :, 0:3].detach().cpu().numpy()+1)/2)
            #plt.show()
            noisy_image = x_start
        
        
        return noisy_image

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2) #.to(self.device)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def calc_schedule_wrong(self):
        betas = torch.linspace(0, 1, self.timesteps).to(self.device)
        alphas = 1 - betas
        sqrt_alphas_cumprod = alphas
        sqrt_one_minus_alphas_cumprod = betas

        posterior_variance = betas 
        sqrt_recip_alphas = alphas

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance
    
    def calc_schedule(self):
        # define beta schedule
        betas = 0
        if self.beta_schedule == "linear":
            betas = self.linear_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "quadratic":
            betas = self.quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            NotImplementedError()

        betas = betas.to(self.device)

        # define alphas
        alphas = 1. - betas
        #print("ALPHAS", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # (alphas, axis=0)
        #print("ALPHAS_CUMPROD", alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        #print("ALPHAS_CUMPROD_PREV", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        #print("SQRT_RECIP_ALPHAS", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        #print("SQRT_ALPHAS_CUMPROD", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        #print("SQRT_ONE_MINUS_ALPHAS_CUMPROD", sqrt_one_minus_alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance

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
        
        #noise = img.clone()
        
        #noise = torch.randn_like(img).to(self.device)

        if False:
            noise = torch.rand(img.shape).to(self.device)*2 -1 #orch.tensor.uniform_(noise)
            print(torch.max(noise), torch.min(noise))
            img_noisy = self.q_sample(x_start=img, t=t, noise=noise)
            img_noisy = torch.clip(img_noisy, -1, 1)
            noise = img_noisy - img
        #else:
        #    noise = torch.randn_like(img).to(self.device)
        #    noise -= noise.min(1, keepdim=True)[0]
        #    noise /= noise.max(1, keepdim=True)[0]
        #    img_noisy = self.q_sample(x_start=img, t=t, noise=noise)

        noise, img_noisy = self.getNoiseLike(img, noisy=True, t=t)

        
        #img_noisy -= img_noisy.min(1, keepdim=True)[0]
        #img_noisy /= img_noisy.max(1, keepdim=True)[0]
        
        #print("IMG NOISy", torch.max(img_noisy), torch.min(img_noisy))
        
        img_noisy = self.make_seed(img_noisy)
        if not eval and self.exp.get_from_config('batch_duplication')!=1:
            img_noisy, noise = self.repeatBatch(img_noisy, noise, self.exp.get_from_config('batch_duplication'))
        data_noisy = (id, img_noisy, img_noisy)

        data = {'id': id, 'image': img_noisy, 'label': img_noisy, 'noise': label}

        return data

    def get_outputs(self, data, full_img=False, t=0, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        t = torch.tensor(t/self.timesteps).to(self.device) # TODO: torch.tensor((t+1)/self.timesteps).to(self.device)
        #print("T", t)
        id, inputs, targets = data['id'], data['image'], data['label']
        if self.exp.model_state == "train":
            #print("TTTTTTT<-------------------", t.shape)
            t = t.repeat(self.exp.get_from_config('batch_duplication'))
        
        if isinstance(self.model, list): # TODO: Add support for > batch size one
            if torch.numel(t) > 1:
                model_id = math.floor(((t[0]-0.0000001) * self.timesteps) / (self.timesteps / len(self.model)))   
                #print(model_id)
            else:
                model_id = math.floor(((t-0.0000001) * self.timesteps) / (self.timesteps / len(self.model)))  
            outputs = self.model[model_id](inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=t, epoch=self.exp.currentStep)
        else:
            outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=t, epoch=self.exp.currentStep)
        
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        #return outputs[..., 0:self.output_channels], targets
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

    def rescale_image(self, data):
        id, img, label = data['id'], data['image'], data['label']

        random_fac = random.uniform(0.5,1)

        img = img.transpose(1,3)
        size = (int(img.shape[2]*random_fac), int(img.shape[3]*random_fac)) 

        img = F.interpolate(img, size=size, mode='bilinear')
        img = img.transpose(1,3)
        #print("here", img.shape)

        return id, img, label

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        if isinstance(self.model, list):
            rang = int((self.timesteps / len(self.model)) * np.random.randint(0, len(self.model)))
            t = torch.randint(1, int(self.timesteps / len(self.model)), (data[1].shape[0],), device=self.exp.get_from_config(tag="device")).long()
            t = torch.add(t, rang)
            #print(t)
        else:
            t = torch.randint(0, self.timesteps, (data[1].shape[0],), device=self.exp.get_from_config(tag="device")).long()


        real_img = data['image'].to(self.device).to(torch.float)

        # Rescale img factor x
        #data = self.rescale_image(data)
        
        data = self.prepare_data(data, t) #data, noise, label = self.prepare_data
        id, img, _, noise, label = data['id'], data['image'], data['label'], data['noise'], data['label']
        outputs, _ = self.get_outputs(data, t=t)

        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                self.optimizer[m].zero_grad()
        else:
            self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}


        if False:
            #outputs = outputs
            #print(outputs.shape)
            outputs_fourier = torch.fft.fft2(outputs.transpose(1, 3))
            #print(outputs.shape, outputs_fourier.shape)
            #noise = noise
            noise_fourier = torch.fft.fft2(noise.transpose(1, 3))
            loss = F.l1_loss(outputs_fourier, noise_fourier, reduction='none')

            # WEIGHTING
            x_count = torch.linspace(0.1, 1, loss.shape[2]).expand(1, 1, loss.shape[2], loss.shape[3]).to(self.device)#.transpose(1,3)
            y_count = torch.transpose(x_count, 2, 3)
            #print(x_count.shape(),y_count.shape())
            alive = 1-torch.maximum(y_count, x_count) #x_count[x_count > y_count] = y_count
            alive = alive * alive

            #plt.imshow(alive[0, 0, :, :].detach().cpu().numpy())
            #plt.show()
            #exit()
            #print(loss.shape)
            loss = torch.mean(loss * alive)
            loss2 = F.l1_loss(outputs, noise)
            loss3 = F.mse_loss(outputs, noise)
            #print(loss, loss2, loss3)

            
            loss = loss + loss2 + loss3 #+ (noise - outputs).square().sum(dim=(1, 2, 3)).mean(dim=0)#

        else:
            #loss = F.l1_loss(outputs, noise)
        #loss = loss * 
        #loss = F.smooth_l1_loss(outputs, noise)

            #print("TENSOR TYPE", noise.type())
            if True:
                #loss = F.mse_loss(outputs, noise)
                
                #loss = F.l1_loss(outputs, noise) + F.mse_loss(outputs, noise) #(noise - outputs).square().sum(dim=(1, 2, 3)).mean(dim=0)# + 
            
                 # USE Variance to balance loss, increase importance of loss when variance is high
                if False:
                    #print(outputs.shape)
                    split = torch.split(outputs, split_size_or_sections=outputs.shape[0]//3, dim=0)
                    #print(split)
                    split = torch.stack(split, 0)
                    #print(split.shape)
                    mean = torch.sum(split, axis=0) / split.shape[0]
                    stdd = 0
                    for id in range(split.shape[0]):
                        img = split[id] - mean
                        img = img*img
                        stdd = stdd + img
                    stdd = stdd / split.shape[0]
                    stdd = torch.clip(stdd, 0, 1)
                    stdd = torch.sqrt(stdd)
                    stdd = torch.cat((stdd, stdd, stdd), 0)
                    #print(F.mse_loss(outputs, noise, reduction='none').shape, stdd.shape)
                    loss = torch.mean(F.mse_loss(outputs, noise, reduction='none') * stdd)
                else:
                    
                    #outputs_fft = torch.fft.fft2(outputs.transpose(1, 3))[..., :20, :20]
                    #noise_fft = torch.fft.fft2(noise.transpose(1, 3))[..., :20, :20]

                    #x_count = torch.linspace(1, outputs_fft.real.shape[2]*2-1, outputs_fft.real.shape[2]).expand(1, 1, outputs_fft.real.shape[2], outputs_fft.real.shape[3]).to(self.device)
                    #y_count = torch.transpose(x_count, 2, 3)
                    #alive = 1/torch.maximum(y_count, x_count) #x_count[x_count > y_count] = y_count
                    #plt.imshow(outputs_fft[0, 0, :, :].real.detach().cpu().numpy())
                    #plt.show()

                    #print(outputs.shape, img.shape)
                    #ct = real_img[..., 0:self.input_channels]
                    #plt.imshow(outputs[0, :, :, 0:3].real.detach().cpu().numpy())
                    #plt.show()
                    #loss = F.l1_loss(outputs, ct) + F.mse_loss(outputs, ct)
                    #loss = F.mse_loss(outputs, noise) + F.l1_loss(outputs, noise)

                    loss = 0


                    if True:
                        # Transform outputs
                        outputs = outputs.transpose(1,3)
                        outputs = torch.fft.fft2(outputs, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
                        outputs = torch.fft.fftshift(outputs, dim=(2,3))

                        noise = noise.transpose(1,3) #[:, :, :, 0:self.exp.get_from_config(tag="input_channels")]
                        noise = torch.fft.fft2(noise, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
                        noise = torch.fft.fftshift(noise, dim=(2,3))

                        #plt.imshow(img.transpose(1,3)[0, :, :, 0:3].real.detach().cpu().numpy()*100)
                        #plt.show()

                        #plt.imshow(outputs.transpose(1,3)[0, :, :, 0:3].real.detach().cpu().numpy()*100)
                        #plt.show()


                        #real_img = real_img.transpose(1,3)
                        #real_img = torch.fft.fft2(real_img, norm="forward")
                        #real_img = torch.fft.fftshift(real_img, dim=(2,3))

                        num_el = t.clone()

                        for b in range(outputs.shape[0]):

                            patch = int(t[b]+1)*2
                            half_x = outputs.shape[2]//2 - patch//2
                            half_y = outputs.shape[3]//2 - patch//2
                            x_temp = outputs[b, :, half_x:half_x+patch, half_y:half_y+patch].clone()
                            outputs[b, :, :, :] = 0
                            outputs[b, :, half_x:half_x+patch, half_y:half_y+patch] = x_temp
                            outputs[b, :, half_x+1:half_x+patch-1, half_y+1:half_y+patch-1] = 0

                            #plt.imshow(outputs.transpose(1,3)[b, :, :, 0:3].real.detach().cpu().numpy()*100)
                            #plt.show()

                            x_temp = noise[b, :, half_x:half_x+patch, half_y:half_y+patch].clone()
                            noise[b, :, :, :] = 0
                            noise[b, :, half_x:half_x+patch, half_y:half_y+patch] = x_temp
                            noise[b, :, half_x+1:half_x+patch-1, half_y+1:half_y+patch-1] = 0

                            #num_el[b] = torch.numel(real_img[real_img!=0])

                            #loss = loss + F.l1_loss(outputs, real_img)

                            #plt.imshow(real_img.transpose(1,3)[b, :, :, 0:3].real.detach().cpu().numpy()*100)
                            #plt.show()
                        
                        noise = torch.fft.fftshift(noise, dim=(2,3))
                        noise = torch.fft.ifft2(noise, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
                        noise = noise.transpose(1,3)

                        outputs = torch.fft.fftshift(outputs, dim=(2,3))
                        outputs = torch.fft.ifft2(outputs, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
                        outputs = outputs.transpose(1,3)

                        #print(noise.shape, outputs.shape)
                        #print(img[..., 0:self.exp.get_from_config(tag="input_channels")].shape, outputs.real.shape)
                        loss = F.mse_loss(real_img, img[..., 0:self.exp.get_from_config(tag="input_channels")] - outputs.real) #+ F.l1_loss(noise.real, outputs.real) 


                        #loss = torch.mean(torch.mean(F.mse_loss(outputs.real, real_img.real, reduction='none'), dim=(1, 2, 3))*num_el + torch.mean(F.mse_loss(outputs.imag, real_img.imag, reduction='none'), dim=(1, 2, 3))*num_el)

                        if False:
                            outputs = torch.fft.ifftshift(outputs, dim=(2,3))
                            outputs = torch.fft.ifft2(outputs, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"
                            
                            

                            real_img = torch.fft.ifftshift(real_img, dim=(2,3))
                            real_img = torch.fft.ifft2(real_img, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"

                            #normalize = torch.sum(torch.abs(real_img))
                            outputs = outputs.transpose(1,3)
                            real_img = real_img.transpose(1,3)


                            loss = torch.mean(F.mse_loss(outputs.real, real_img.real, reduction='none'), dim=(1, 2, 3))

                            loss = torch.mean(loss)
                    else:
                        loss = torch.mean(F.mse_loss(outputs.real, real_img.real, reduction='none'), dim=(1, 2, 3))
                        loss = torch.mean(loss)

                    #plt.imshow(outputs[0, :, :, 0:3].real.detach().cpu().numpy())
                    #plt.show()

                    #plt.imshow(real_img[0, :, :, 0:3].real.detach().cpu().numpy())
                    #plt.show()
                    
                    
                    #torch.mean(F.l1_loss(outputs_fft.real, noise_fft.real, reduction="none")*alive) #+ 10*torch.mean(F.l1_loss(outputs_fft.imag, noise_fft.imag, reduction="none")*alive)  #F.l1_loss(outputs, noise) # + 0.1*
            else:
                #print("REAL", real_img.shape, outputs.shape)
                model_noise = self.q_sample(x_start=real_img, t=t, noise=outputs)
                #print(model_noise.shape, img.shape)

                if False:
                    outputs_fourier = torch.fft.fft2(model_noise.transpose(1, 3))
                    noise_fourier = torch.fft.fft2(img[..., 0:self.exp.get_from_config(tag="input_channels")].transpose(1, 3))
                    loss = F.l1_loss(outputs_fourier, noise_fourier, reduction='none') + F.l1_loss(outputs_fourier, noise_fourier, reduction='none')

                    # WEIGHTING
                    #x_count = torch.linspace(0.1, 1, loss.shape[2]).expand(1, 1, loss.shape[2], loss.shape[3]).to(self.device)#.transpose(1,3)
                    x_count = torch.linspace(1, loss.shape[2]*2-1, loss.shape[2]).expand(1, 1, loss.shape[2], loss.shape[3]).to(self.device)
                    y_count = torch.transpose(x_count, 2, 3)
                    alive = 1/torch.maximum(y_count, x_count) #x_count[x_count > y_count] = y_count
                    alive = alive 

                    loss = torch.mean(loss * alive)
                #loss2 = F.l1_loss(outputs, noise)
                #loss3 = F.mse_loss(outputs, noise)

                loss = F.l1_loss(model_noise, img[..., 0:self.exp.get_from_config(tag="input_channels")], reduction='none') + F.mse_loss(model_noise, img[..., 0:self.exp.get_from_config(tag="input_channels")], reduction='none')
                #print(loss.shape)
                loss, _ = torch.sort(torch.flatten(loss))
                #print(loss.shape)

                #loss = loss[int(len(loss)/10):int(-len(loss)/10)]
                loss = loss * torch.linspace(1, 0.1, loss.shape[0]).to(self.device)
                #print(loss.shape)
                loss = torch.mean(loss)

        #loss = torch.mean(torch.sum(torch.square(noise - outputs), dim=(1, 2, 3)) , dim=0)
        
        #loss = (noise - outputs).square().sum(dim=(1, 2, 3)).mean(dim=0)
        
        #print("LOSS", loss)
        loss_ret[0] = loss

        
        #if len(outputs.shape) == 5:
        #    for m in range(outputs.shape[-1]):
        #        loss_loc = loss_f(outputs[..., m], targets[...])
        #        loss = loss + loss_loc
        #        loss_ret[m] = loss_loc.item()
        #else:
        #    for m in range(outputs.shape[-1]):
        #        if 1 in targets[..., m]:
        #            loss_loc = loss_f(outputs[..., m], targets[..., m])
        #            loss = loss + loss_loc
        #            loss_ret[m] = loss_loc.item()

        loss.backward()
        if isinstance(self.optimizer, list): 
            for m in range(len(self.optimizer)):
                #torch.nn.utils.clip_grad_norm_(self.model[m].parameters(), 0.1)
                self.optimizer[m].step()
        else:
            self.optimizer.step()
        if isinstance(self.scheduler, list): 
            for m in range(len(self.scheduler)):
                self.scheduler[m].step()
        else:
            self.scheduler.step()
        return loss_ret

    def getNoiseLike(self, img, noisy=False, t=0):

        def getNoise():
            rnd = torch.randn_like(img).to(self.device).to(torch.float) /5
            # Range 0,1
            #rmax, rmin = torch.max(rnd), torch.min(rnd)
            #rnd = ((rnd - rmin) / (rmax - rmin))
            #rnd = rnd*2 -1
            #rnd = rnd*5
            return rnd #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) 
        #noise = torch.rand(img.shape).to(self.device)*2 -1 #orch.tensor.uniform_(noise)
        
        #noise = torch.randn_like(img).to(self.device)
        #noise -= noise.min(0, keepdim=True)[0]
        #noise /= noise.max(0, keepdim=True)[0]
        #noise = noise*2 -1
        
        

        #äprint("NOISE", noise[0].min(), noise[0].max())
        
        if noisy:
            #noise = torch.randn_like(img).to(self.device) #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) #
            noise = getNoise()
            img_noisy = self.q_sample(x_start=img, t=t, noise=noise)
            #img_noisy = torch.clip(img_noisy, -1, 1)
            #noise = img_noisy - img
            img_noisy = img_noisy.to(self.device)
            noise = noise
        else:
            #noise = torch.randn_like(img).to(self.device) #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) #
            noise = getNoise()
            img_noisy = 0

        return noise.to(self.device), img_noisy

    @torch.no_grad()
    def p_sample(self, output, x, t, t_index):

        output = output.transpose(1,3)
        output = torch.fft.fft2(output, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
        output = torch.fft.fftshift(output, dim=(2,3))

        #plt.imshow((x[0, :, :, 0:3].real.detach().cpu().numpy()+1)/2)
        #plt.show()

        x = x.transpose(1,3)
        x = torch.fft.fft2(x, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
        x = torch.fft.fftshift(x, dim=(2,3))

        for b in range(output.shape[0]):
            patch = int(t[b]+1)*2 +1
            half_x = output.shape[2]//2 - patch//2
            half_y = output.shape[3]//2 - patch//2


            x_clone = x.clone()

            #x_temp = x[b, :, half_x:half_x+patch, half_y:half_y+patch].clone()
            #x[b, :, :, :] = 0
            #x[b, :, half_x:half_x+patch, half_y:half_y+patch] = x_temp

            output[b, :, half_x+1:half_x+patch-1, half_y+1:half_y+patch-1] = 0

            #x[b, :, half_x:half_x+patch, half_y:half_y+patch] = output[b, :, half_x:half_x+patch, half_y:half_y+patch]
            #x[b, :, half_x+1:half_x+patch-1, half_y+1:half_y+patch-1] = x_clone[b, :, half_x+1:half_x+patch-1, half_y+1:half_y+patch-1]

        #plt.imshow(x.transpose(1,3)[0, :, :, 0:3].real.detach().cpu().numpy()*100)
        #plt.show()

        x = torch.fft.ifftshift(x, dim=(2,3))
        x = torch.fft.ifft2(x, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"

        output = torch.fft.ifftshift(output, dim=(2,3))
        output = torch.fft.ifft2(output, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"

        x = x - output

        x = x.transpose(1,3)





        return x

        if False:
            betas_t = self.extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

            # Equation 11 in the paper
            # Use our model (noise predictor) to predict the mean
            #print(x.shape, betas_t.shape, output.shape, sqrt_one_minus_alphas_cumprod_t.shape)
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * output / sqrt_one_minus_alphas_cumprod_t
            )
            # return output
            if t_index == 0:
                return model_mean
            else:
                posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                noise, _ = self.getNoiseLike(x) #torch.randn_like(x)
                # Algorithm 2 line 4:
                return model_mean + torch.sqrt(posterior_variance_t) * noise

    def intermediate_evaluation(self, dataloader, epoch):
        if not self.averages:
            self.calculateAverages()
        self.exp.set_model_state("test")
        #if epoch % 5 == 0:
        #    self.test_fid()
        self.test()
        self.exp.set_model_state("train")

    def generateSamples(self, samples:int=1, normal=True):
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        #diceLoss = DiceLoss(useSigmoid=useSigmoid)
        self.exp.set_model_state("test")
        self.test(tag="extra", samples=samples, extra=False, normal=normal)

        #return loss_log

    @torch.no_grad()
    def test_fid(self, tag:str='0', samples:int=1, extra:bool=False, optimized=False, saveImg=False, **kwargs):
        if not self.averages:
            self.calculateAverages()
        size = self.exp.get_from_config('input_size')

        if samples < 2: samples = 2

        # Generate samples
        noise, _ = self.getNoiseLike(torch.zeros((samples, size[0], size[1], self.exp.get_from_config('input_channels'))))
        img = self.make_seed(noise)

        if optimized:
            for i in tqdm(range(samples)):
                sample = self.optimizedImageGeneration(size)
                img[i, ...] = sample
                #img[i*4+1, ...] = sample
                #img[i*4+2, ...] = sample
                #img[i*4+3, ...] = sample
                if saveImg:
                    # SAVE IMAGE
                    name = random.randint(0, 2000000000)#choices(string.ascii_lowercase, k=10)
                    path = os.path.join(self.exp.get_from_config('model_path'), "models", "epoch_" + str(self.exp.currentStep-1), "Generated")
                    #print(path)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path = os.path.join(path, str(name)+ ".jpg")
                    cv2.imwrite(path, cv2.cvtColor(torch.clip(((sample[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu()+1)/2)*255, 0, 255).to(torch.uint8).numpy(), cv2.COLOR_RGB2BGR))

            
        else:
            #noise, _ = self.getNoiseLike(torch.zeros((samples, int(size[0]), int(size[1]), self.exp.get_from_config('input_channels'))))
            #img = self.make_seed(noise)
            print(img.shape)
            for step in tqdm(reversed(range(1))): #self.timesteps
                #for i in range(2):
                t = torch.full((samples,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                #print("NOISE HERE", torch.max(img), torch.min(img))
                output, _ = self.get_outputs(img_p, t = step)
                img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])

        # Compose images
        factor = 4
        composition = np.zeros((factor*size[0], factor*size[1], self.exp.get_from_config('input_channels')))
        imgs = (img.detach().cpu().numpy()+1)/2
        for i, b in enumerate(range(img.shape[0])):
            x = b % 4
            y = int(math.floor(b//factor))
            composition[x*size[0]:(x+1)*size[0], y*size[0]:(y+1)*size[0], 0:self.exp.get_from_config('input_channels')] = imgs[b, ... , 0:self.exp.get_from_config('input_channels')] 
            if i == (factor*factor -1): break
        self.exp.write_img("Composition", composition, self.exp.currentStep, normalize=True) #context={'Image':s}, 

        print(img.transpose(1,3).shape)

        sample = (((img[..., 0:self.exp.get_from_config('input_channels')].transpose(1,3).detach().cpu()+1)/2)*256) #
        sample = torch.clip(sample, 0, 255).to(torch.uint8)
        print(torch.min(sample), torch.max(sample))
        self.exp.getFID().update(sample, real=False)
        fid_score = self.exp.fid.compute()
        print("FID: ", fid_score)
        self.exp.write_scalar('FID', fid_score, self.exp.currentStep)

    def calculateFID_fromFiles(self, samples):
        path = os.path.join(self.exp.get_from_config('model_path'), "models", "epoch_" + str(self.exp.currentStep-1), "Generated")
        #path = r"/home/jkalkhof_locale/Documents/GitHub/vnca2/Synth/"
        imgs = None
        for i, file in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            img = torch.unsqueeze(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(path, file)), cv2.COLOR_BGR2RGB)).to(torch.uint8), dim=0)

            if imgs is None:
                imgs = img #expand_dims(img, axis=0)
            else:
                imgs = torch.cat((imgs, img), dim=0)
            # Only use 2048 samples
            if i == 2047: break

        #    if i % samples == 0 and i != 0:
        #print(imgs.transpose(1,3).shape)
        self.exp.getKID().update(imgs.transpose(1,3), real=False)
        print("KID: ", self.exp.kid.compute())
        self.exp.getFID().update(imgs.transpose(1,3), real=False)
        print("FID: ", self.exp.fid.compute())
        imgs = None


    def calculateSDV(self, image): 
        total_noise_std_dev = 0

        for channel in range(3):
            # Extract the specific color channel
            single_channel = image[:, :, channel]

            # Calculate the difference between neighboring pixels horizontally
            diff_horizontal = single_channel[:, 1:] - single_channel[:, :-1]

            # Calculate the difference between neighboring pixels vertically
            diff_vertical = single_channel[1:, :] - single_channel[:-1, :]

            # Combine the horizontal and vertical differences
            diff_combined = np.hstack((diff_horizontal.flatten(), diff_vertical.flatten()))

            # Estimate the standard deviation of the differences for this channel
            noise_std_dev = np.std(diff_combined)

            # Add the standard deviation for this channel to the total
            total_noise_std_dev += noise_std_dev

            #print(f'Channel {channel}: Estimated Gaussian noise standard deviation: {noise_std_dev}')
        
        return total_noise_std_dev
    
    def calculateAverages(self):
        self.exp.set_model_state('test')
        dataset = self.exp.dataset
        size = self.exp.get_from_config('input_size')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        list_key = []
        list_val = []
        for i, data in tqdm(enumerate(dataloader)):  
            _, img, _ = data['id'], data['image'], data['label']
            img = img.to(self.device)

            for step in range(self.timesteps):
                t = torch.full((1,), step, device=self.exp.get_from_config(tag="device"), dtype=torch.long)
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))          
                img_noisy = self.q_sample(x_start=img, t=t, noise=noise).detach().cpu().numpy()
                
                image = img_noisy[0, ...]
                total_noise_std_dev = self.calculateSDV(image)
                list_key.append(step)
                list_val.append(total_noise_std_dev)
                #dic[ts] = total_noise_std_dev

            if i > 100:
                #import seaborn as sns
                #sns.scatterplot(x=list_key, y=list_val)
                #plt.show()

                # Calculate average of each entry
                from collections import defaultdict
                id_values = defaultdict(lambda: {'sum': 0, 'count': 0})

                for id, value in zip(list_key, list_val):
                    id_values[id]['sum'] += value
                    id_values[id]['count'] += 1

                averages = []
                for id, data in id_values.items():
                    avg = data['sum'] / data['count']
                    averages.append((id, avg))

                #print(averages)
                self.averages = averages
                break

    def optimizedImageGeneration(self, size):
        if not self.averages:
            self.calculateAverages()
        noise, _ = self.getNoiseLike(torch.zeros((1, int(size[0]), int(size[1]), self.exp.get_from_config('input_channels'))))
        img = self.make_seed(noise)

        currentTimestep = range(self.timesteps)[-1]
        pbar = tqdm(total=self.timesteps, leave=False)
        while True:#
        #for step in tqdm(reversed(range(self.timesteps))):
            step = int(currentTimestep)#/self.timesteps
            #print(step)
            t = torch.full((1,), step, device=self.device, dtype=torch.long)
            img_p = 0, img, 0
            #print("NOISE HERE", torch.max(img), torch.min(img))
            output, _ = self.get_outputs(img_p, t = step)
            img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
            img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])

            # Find closest timestep
            min_difference = 400
            min_pos = 400
            image = img[0, ..., 0:3].detach().cpu().numpy()
            total_noise_std_dev = self.calculateSDV(image)
            for i, l in enumerate(self.averages):
                st, val = l
                if abs(total_noise_std_dev-val) < min_difference:
                    min_difference = abs(total_noise_std_dev-val)
                    min_pos = i
            
            if step == 0:
                break

            currentTimestep, _ = self.averages[min_pos]
            if step < 40:# and currentTimestep < step-1:
                currentTimestep = step-1

            pbar.n = self.timesteps-currentTimestep
            pbar.refresh()

        return img
            


    @torch.no_grad()
    def test(self, tag='0', samples=3, extra=False, normal=True, **kwargs):

        # Generate sample
        size = self.exp.get_from_config('input_size')
        
        dataset = self.exp.dataset
        self.exp.set_model_state('train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        #self.test_fid()
        #noise = torch.randn_like(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels')))).to(self.device)

        #self.timesteps = 300



        if normal:
            for s in range(samples):
                if True:

                    noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                    #noise = torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels')))
                    if False:
                        for i, data in enumerate(dataloader):  
                            
                            # forward step to 80%
                            _, img, _ = data['id'], data['image'], data['label']
                            #img, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                            if i == s:
                                break
                    else:
                        img, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))

                    #noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                    
    
                    #plt.imshow((img[0, :, :, 0:3].real.detach().cpu().numpy()+1)/2)
                    #plt.show()

                    img = self.make_seed(self.q_sample(img.to(self.device), torch.full((1,), 0, device=self.device, dtype=torch.long), noise).to(self.device))
                    
                    #print("TIMESTEPS", self.timesteps)
                    for step in range(0, self.timesteps, 1): #self.timesteps
                        print("SSTTEEEEEEEEP", step)
                        
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)

                        #img = output
                        #plt.imshow(output[0, :, :, 0:3].real.detach().cpu().numpy())
                        #plt.show()

                        #t = torch.full((1,), 0, device=self.device, dtype=torch.long)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img)#img[..., self.exp.get_from_config('input_channels'):self.exp.get_from_config('output_channels')])
                    self.exp.write_img(tag, (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                else:
                    size_loc = deepcopy(size)
                    size_loc[0] *= 1.23 #1.23
                    size_loc[1] *= 1#2
                    img = self.optimizedImageGeneration(size=size_loc)

                    self.exp.write_img(tag, (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                    
            
            #/2 +0.5

        if False:
            # Extra steps
            for step in reversed(range(int(self.timesteps/2))):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                output, _ = self.get_outputs(img_p, step)
                img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img("extra_steps 50%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep, normalize=True)
            # Extra steps
            for step in reversed(range(int(self.timesteps/2))):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                output, _ = self.get_outputs(img_p, step)
                img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img("extra_steps 100%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep, normalize=True)
        # For very long runs
        if False:
            for i in range(5):
                for step in reversed(range(int(self.timesteps))):
                    t = torch.full((1,), step, device=self.device, dtype=torch.long)
                    img_p = 0, img, 0
                    output, _ = self.get_outputs(img_p, step)
                    img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                    img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img("extra_steps" + str(100 + (i+1)*100) + "%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep)
        if extra:
            if False:
                for i, data in enumerate(dataloader):   
                    noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))

                    if i < 1175:
                        continue

                    for s in range(samples):
                        _, img, _ = data['id'], data['image'], data['label']
                        img = img.to(self.device)

                        start_x = 32
                        end_x = 48
                        start_y = 29
                        end_y = 45
                        img_tosave = torch.clone(img)
                        img_tosave[:, start_x:end_x, start_y:end_y, :] = noise[:, start_x:end_x, start_y:end_y, :]/5

                        img[:, start_x:end_x, start_y:end_y, :] = noise[:, start_x:end_x, start_y:end_y, :]
                        img_copy = torch.clone(img)
                        img = self.make_seed(img)
                        self.exp.write_img("Before Inpainting", (img_tosave[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, normalize=True) #/2+0.5 #{'Image':s}



                        # Inpainting
                        noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                        for step in tqdm(reversed(range(self.timesteps))):
                            #for i in range(2):
                            t = torch.full((1,), step, device=self.device, dtype=torch.long)
                            img_p = 0, img, 0
                            #print("NOISE HERE", torch.max(img), torch.min(img))
                            output, _ = self.get_outputs(img_p, t = step)
                            img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                            img_loc = torch.clone(img_copy)
                            img_loc[:, start_x:end_x, start_y:end_y, :] = img[:, start_x:end_x, start_y:end_y, :]
                            img = img_loc
                            img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                        self.exp.write_img("After Inpainting", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, normalize=True, context={'Image':s},) #/2+0.5 #{'Image':s}
                    break

            # Recover
            if False:   
                for i, data in enumerate(dataloader):   
                    noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))

                    if i < 1:
                        continue

                    # forward step to 80%
                    _, img, _ = data['id'], data['image'], data['label']
                    img = img.to(self.device)

                    #RESCALE IMAGE
                    img = img.transpose(1,3)
                    size = (int(img.shape[2]), int(img.shape[3])) 

                    img = F.interpolate(img, size=size, mode='bilinear')
                    img = img.transpose(1,3)

                    t = torch.full((1,), self.timesteps//1*1, device=self.exp.get_from_config(tag="device"), dtype=torch.long)
                    img = self.q_sample(x_start=img, t=t, noise=noise)
                    img = self.make_seed(img)
                    for step in tqdm(reversed(range(self.timesteps//1*1))):
                        step = step
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                    self.exp.write_img("Recover", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                    #/2 +0.5
                    break

            # Superresolution
            if True:
                print("SUPERSAMPLE")
                for i, data in enumerate(dataloader):  
                    
                    
                    sup_res = 1.5 #1.44
                    if False:
                        if i < 42:
                            continue



                        # forward step to 80%
                        _, img, _ = data['id'], data['image'], data['label']
                    else:
                        img = cv2.imread("/home/jkalkhof_locale/Downloads/565005978_2.png")
                        img = (torch.unsqueeze(torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float64), dim=0)/128)-1
                    img = img.to(self.device)

                    #RESCALE IMAGE
                    print(img.shape)
                    self.exp.write_img("Before Superresolution", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, normalize=True) #/2+0.5 #{'Image':s}
            
                    for l in range(1):
                        print(img.shape)
                        img = img.transpose(1,3)
                        size_sup = (int(img.shape[2]*sup_res), int(img.shape[3]*sup_res)) 
                        img = F.interpolate(img, size=size_sup, mode='nearest')
                        img = img.transpose(1,3)

                        print(img.shape)
                        noise, _ = self.getNoiseLike(torch.zeros((1, img.shape[1], img.shape[2], self.exp.get_from_config('input_channels'))))
                        print(noise.shape)

                        t = torch.full((1,), self.timesteps//int(14*(l+1)), device=self.exp.get_from_config(tag="device"), dtype=torch.long)
                        img = self.q_sample(x_start=img, t=t, noise=noise)
                        img = self.make_seed(img)
                        for step in tqdm(reversed(range(self.timesteps//int(14*(l+1))))):
                            for i in range(1):
                                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                                img_p = 0, img, 0
                                #print("NOISE HERE", torch.max(img), torch.min(img))
                                output, _ = self.get_outputs(img_p, t = step)
                                img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                        img = img[0:1, ..., 0:self.exp.get_from_config('input_channels')]
                    self.exp.write_img("Superresolution", (img[0].detach().cpu().numpy()+1)/2, self.exp.currentStep, normalize=True) #/2+0.5 #{'Image':s}
                    #/2 +0.5
                    break

            if False:
                for s in range(samples):
                    bigger = 6
                    noise, _ = self.getNoiseLike(torch.zeros((1, int(size[0]*bigger), int(size[1]*bigger), self.exp.get_from_config('input_channels'))))
                    img = self.make_seed(noise)
                    for step in tqdm(reversed(range(self.timesteps))):
                        for i in range(1):
                            t = torch.full((1,), step, device=self.device, dtype=torch.long)
                            img_p = 0, img, 0
                            #print("NOISE HERE", torch.max(img), torch.min(img))
                            output, _ = self.get_outputs(img_p, t = step)
                            img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                            img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                    self.exp.write_img("bigger_size", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}

                    sup_res = 1.5
                    noise, _ = self.getNoiseLike(torch.zeros((1, int(size[0]*sup_res*bigger), int(size[1]*sup_res*bigger), self.exp.get_from_config('input_channels'))))

                    # forward step to 80%
                    img = img[..., 0:self.exp.get_from_config('input_channels')]

                    #RESCALE IMAGE
                    img = img.transpose(1,3)
                    size_sup = (int(img.shape[2]*sup_res), int(img.shape[3]*sup_res)) 

                    img = F.interpolate(img, size=size_sup, mode='bilinear')
                    img = img.transpose(1,3)

                    t = torch.full((1,), self.timesteps//10, device=self.exp.get_from_config(tag="device"), dtype=torch.long)
                    img = self.q_sample(x_start=img, t=t, noise=noise)
                    img = self.make_seed(img)
                    for step in tqdm(reversed(range(self.timesteps//10))):
                        for i in range(1):
                            t = torch.full((1,), step, device=self.device, dtype=torch.long)
                            img_p = 0, img, 0
                            #print("NOISE HERE", torch.max(img), torch.min(img))
                            output, _ = self.get_outputs(img_p, t = step)
                            img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                            img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                    self.exp.write_img("bigger_size_Sup", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                    #/2 +0.5

            for s in range(samples):
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                img = self.make_seed(noise)
                for step in tqdm(reversed(range(self.timesteps))):
                    for i in range(1):
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img(tag + "doubleSteps", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                #/2 +0.5

                sup_res = 3
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0]*sup_res, size[1]*sup_res, self.exp.get_from_config('input_channels'))))

                # forward step to 80%
                img = img[..., 0:self.exp.get_from_config('input_channels')]

                #RESCALE IMAGE
                img = img.transpose(1,3)
                size_sup = (int(img.shape[2]*sup_res), int(img.shape[3]*sup_res)) 

                img = F.interpolate(img, size=size_sup, mode='bilinear')
                img = img.transpose(1,3)

                t = torch.full((1,), self.timesteps//10, device=self.exp.get_from_config(tag="device"), dtype=torch.long)
                img = self.q_sample(x_start=img, t=t, noise=noise)
                img = self.make_seed(img)
                for step in tqdm(reversed(range(self.timesteps//10))):
                    for i in range(1):
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img("Generate_Sup", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                #/2 +0.5

