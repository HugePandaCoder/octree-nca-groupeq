import torch
import numpy as np
from src.utils.helper import convert_image, dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent_NCA import Agent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from src.utils.helper import dump_pickle_file, load_pickle_file
import os
from src.losses.LossFunctions import DiceLoss, DiceBCELoss
import copy
import random
from scipy.ndimage import zoom
import torchio as tio
import matplotlib.pyplot as plt
import math

class Agent_NCA_3dOptVRAM_EfficientInference(Agent):
    def initialize(self):
        super().initialize()
        self.stacked_models = self.exp.get_from_config('stacked_models')
        self.scaling_factor = self.exp.get_from_config('scaling_factor')

    #def make_seed(self, img):
    #    r"""Create a seed for the NCA - TODO: Currently only 0 input
    #        Args:
    #            shape ([int, int]): height, width shape
    #            n_channels (int): Number of channels
    #    """###

        # TODO: Initialize with model on smaller scale
#        seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
#        seed[..., :img.shape[3]] = img
#        return seed       

    def prepare_data(self, data, eval=False):
        r"""Prepare the data to be used with the model
            Args:
                data (int, tensor, tensor): identity, image, target mask
            Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            if self.exp.get_from_config('Persistence'):
                inputs = self.pool.getFromPool(inputs, id, self.device)
            inputs, targets = self.repeatBatch(inputs, targets, self.exp.get_from_config('repeat_factor'))
        return id, inputs, targets

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            Args:
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            Returns:
                loss item
        """
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        #print("BBBBBB")
        #print(outputs.shape)
        #print(targets.shape)
        for m in range(self.exp.get_from_config('train_model')+1):
            self.optimizer[m].zero_grad()
        #targets = targets.int()
        loss = 0
        loss_ret = {}
        for m in range(outputs.shape[-1]):
            #print(m, targets.shape, outputs.shape)
            if 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
                #if m == 0:
                #    loss_loc = loss_loc * 100
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()

        #if loss > 1.7:
        #    print("SKIP LOSS")
        #    return loss_ret

        if loss != 0:
            loss.backward()
            for m in range(self.exp.get_from_config('train_model')+1):
                self.optimizer[m].step() #self.exp.get_from_config('train_model')
                self.scheduler[m].step()
        return loss_ret

    def get_outputs(self, data, full_img=False):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data

        if len(targets.shape) < 5:
            targets = torch.unsqueeze(targets, 4)

        
        scale_fac = 2

        if self.exp.get_from_config('scale_factor') is not None:
            scale_fac = self.exp.get_from_config('scale_factor')

        down_scaled_size = (int(inputs.shape[1] / 4), int(inputs.shape[2] / 4), int(inputs.shape[3] / 4))
        #inputs_loc = self.resize4d(inputs.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device')) #torch.from_numpy(zoom(inputs.cpu(), (1, 4, 4, 1))).to(self.exp.get_from_config('device'))
        
        avg_pool = torch.nn.MaxPool3d(2, 2, 0)#torch.nn.MaxPool3d(3, 2, 1)#torch.nn.MaxPool3d(2, 2, 0)
        max_pool = torch.nn.MaxPool3d(2, 2, 0)#torch.nn.MaxPool3d(3, 2, 1)#torch.nn.MaxPool3d(2, 2, 0)
        
        #print(inputs.shape)
        #if scale_fac == 4:
        #    inputs_loc = avg_pool(avg_pool(inputs.transpose(1,4)))
        #if scale_fac == 8:
        #    inputs_loc = avg_pool(avg_pool(avg_pool(inputs.transpose(1,4))))
        #else:
        #    inputs_loc = avg_pool(inputs.transpose(1,4))
        
        #print(inputs_loc.shape)
        targets_loc = targets #self.resize4d(targets.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device'))#torch.from_numpy(zoom(targets.cpu(), (1, 4, 4, 1))).to(self.exp.get_from_config('device'))

        #current_res = full_res

        # Scale Image to Initial Size
        full_res = inputs
        full_res_gt = targets
        inputs_loc = inputs

        for i in range(self.exp.get_from_config('train_model')*int(math.log2(scale_fac))):
            #print(current_res.shape)
            inputs_loc = inputs_loc.transpose(1,4)
            inputs_loc = avg_pool(inputs_loc)
            inputs_loc = inputs_loc.transpose(1,4)
            targets_loc = targets_loc.transpose(1,4)
            targets_loc = max_pool(targets_loc)
            targets_loc = targets_loc.transpose(1,4)

        #print(inputs_loc.shape)
        #plt.imshow(inputs_loc[0, :, :, 3, 0].detach().cpu().numpy())
        #plt.show()

        input_channel = self.exp.get_from_config('input_channels')

        outputs_img = []
        outputs_targets = []
        
        splitInto=3

        if full_img == True:
            with torch.no_grad():
                for m in range(self.exp.get_from_config('train_model')+1):
                    print("LEVEL: ", m)
                    
                    if m == self.exp.get_from_config('train_model'):
                        if type(self.getInferenceSteps()) is list:
                            stp = self.getInferenceSteps()[m]
                        else:
                            stp = self.getInferenceSteps()
                        #print(stp)
                        outputs = inputs_loc
                        for s in range(stp): 
                            print("LEVEL: ", m, " STEP: ", s)
                            size = torch.tensor(outputs.shape[1:4])
                            print(size)
                            size = torch.floor(size/3)
                            print("SIZE", size)
                            for x in range(splitInto):
                                for y in range(splitInto):
                                    for z in range(splitInto):
                                        #### X
                                        start_x = int(size[0] * x)
                                        if x == splitInto-1:
                                            end_x = int(outputs.shape[1])
                                        else:
                                            end_x = int(size[0] * (x+1))
                                        #### Y
                                        start_y = int(size[1] * y)
                                        if y == splitInto-1:
                                            end_y = int(outputs.shape[2])
                                        else:
                                            end_y = int(size[1] * (y+1))
                                        #### Z
                                        start_z = int(size[2] * z)
                                        if z == splitInto-1:
                                            end_z = int(outputs.shape[3])
                                        else:
                                            end_z = int(size[2] * (z+1))
                                        #print(start_x, end_x, start_y, end_z, start_y, end_z)
                                        #start_x, end_x, start_y, end_z, start_y, end_z = int(start_x.item()), int(end_x.item()), int(start_y.item()), int(end_z.item()), int(start_y.item()), int(end_z.item())
                                        #start_x, end_x, start_y, end_z, start_y, end_z = start_x, end_x, start_y.int(), end_z.int(), start_y.int(), end_z.int()
                                        #print(start_x, end_x, start_y, end_z, start_y, end_z)
                                        outputs[:, start_x:end_x,  start_y:end_y,  start_z:end_z, :] = self.model[m](outputs[:, start_x:end_x,  start_y:end_y,  start_z:end_z, :], steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                        print("MiniStep: ", x, y, z)
                        #plt.imshow(outputs[0,:,:,8,1].detach().cpu() - 20*targets_loc[0,:,:,8,0].detach().cpu())
                        #plt.show()
                    else:
                        #print(self.getInferenceSteps())
                        outputs = inputs_loc
                        for s in range(self.getInferenceSteps()[m]): 
                            print("LEVEL: ", m, " STEP: ", s)
                            outputs = self.model[m](outputs, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        
                        up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')

                        outputs = torch.permute(outputs, (0, 4, 1, 2, 3))

                        outputs = up(outputs)
                        inputs_loc = inputs     
                        outputs = torch.permute(outputs, (0, 2, 3, 4, 1))         
   
                        # NEXT RES
                        next_res = full_res
                        for i in range(self.exp.get_from_config('train_model') - (m +1)):
                            next_res = next_res.transpose(1,4)
                            next_res = avg_pool(next_res)
                            next_res = next_res.transpose(1,4)

                        inputs_loc = torch.concat((next_res[...,:input_channel], outputs[...,input_channel:]), 4)
                        #print(inputs_loc.shape)
                        targets_loc = targets

        #print("TARGETS_LOC3", targets_loc.shape)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc 

    def resize4d(self, img, size=(64,64), factor=4, label=False):
        if label:
            transform = tio.Resize((size[0], size[1], size[2], -1), image_interpolation='NEAREST')#((int(img.shape[1]/factor), int(img.shape[2]/factor), -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], size[2], -1))#((int(img.shape[1]/factor), int(img.shape[2]/factor), -1))
        #for i in range(img.shape[0]):
        img = transform(img)
        return img

    def random_crop(self, img, label, outputs):
        size = self.exp.get_from_config('input_size')[0]
        pos_x = random.randint(0, img.shape[1] - size[0])
        pos_y = random.randint(0, img.shape[2] - size[1])

        transform = tio.Resize((int(img.shape[1]), int(img.shape[2]), -1))
        outputs = transform(outputs.cpu()) 
        outputs = outputs.to(self.exp.get_from_config('device'))
        #outputs = torch.from_numpy(np.resize(outputs.cpu(), (outputs.shape[0], int(outputs.shape[1]*4), int(outputs.shape[2]*4), outputs.shape[3]))).to(self.exp.get_from_config('device'))
        #print(outputs.shape)
        outputs = outputs[:, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]

        #print(pos_x)
        #print(pos_y)
        #print(outputs.shape)

        img = img[:, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]
        label = label[:, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]
        img[:, :, :, self.exp.get_from_config('input_channels'):] = outputs[:, :, :, self.exp.get_from_config('input_channels'):]

        return img, label

    def load_state(self, model_path):
        r"""Load state - Add Pool to state
        """
        # TODO: Load all model scales
        super().load_state(model_path)
        if os.path.exists(os.path.join(model_path, 'pool.pbz2')):
            self.pool = load_compressed_pickle_file(os.path.join(model_path, 'pool.pbz2'))

    def save_state(self, model_path):
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)

        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            torch.save(m.state_dict(), os.path.join(model_path, 'model'+ str(id) +'.pth'))
            torch.save(o.state_dict(), os.path.join(model_path, 'optimizer'+ str(id) +'.pth'))
            torch.save(s.state_dict(), os.path.join(model_path, 'scheduler'+ str(id) +'.pth'))

    def load_state(self, model_path):
        r"""Load state of current model
        """
        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            m.load_state_dict(torch.load(os.path.join(model_path, 'model'+ str(id) +'.pth')))
            o.load_state_dict(torch.load(os.path.join(model_path, 'optimizer'+ str(id) +'.pth')))
            s.load_state_dict(torch.load(os.path.join(model_path, 'scheduler'+ str(id) +'.pth')))