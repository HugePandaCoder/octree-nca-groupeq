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

class Agent_NCA_3dOptVRAM(Agent):
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
        avg_pool = torch.nn.AvgPool3d(3, 2, 1)
        max_pool = torch.nn.MaxPool3d(3, 2, 1)
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
        

        if full_img == True:
            with torch.no_grad():
                for m in range(self.exp.get_from_config('train_model')+1):
                    
                    if m == self.exp.get_from_config('train_model'):
                        outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        #plt.imshow(outputs[0,:,:,8,1].detach().cpu() - 20*targets_loc[0,:,:,8,0].detach().cpu())
                        #plt.show()
                    else:
                        outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        
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

                        inputs_loc = torch.concat((next_res[...,:1], outputs[...,1:]), 4)
                        #print(inputs_loc.shape)
                        targets_loc = targets
        else:

            for m in range(self.exp.get_from_config('train_model')+1): #range(self.exp.get_from_config('train_model')+1):

                if m == self.exp.get_from_config('train_model'):
                    #print("AAAAAA")
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                else:
                    # NEXT RES
                    next_res = full_res
                    for i in range(self.exp.get_from_config('train_model') - (m +1)):
                        next_res = next_res.transpose(1,4)
                        next_res = avg_pool(next_res)
                        next_res = next_res.transpose(1,4)
                    # NEXT RES GT
                    next_res_gt = full_res_gt
                    for i in range(self.exp.get_from_config('train_model') - (m +1)):
                        next_res_gt = next_res_gt.transpose(1,4)
                        next_res_gt = max_pool(next_res_gt)
                        next_res_gt = next_res_gt.transpose(1,4)

                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                    

                    up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')

                    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))

                    # to full_res
                    outputs = up(outputs)
   
                    outputs = torch.permute(outputs, (0, 2, 3, 4, 1))        
   
                    inputs_loc = torch.concat((next_res[...,:1], outputs[...,1:]), 4)

                    
                    targets_loc = next_res_gt

                    size = self.exp.get_from_config('input_size')[0]

                    inputs_loc_temp = inputs_loc
                    targets_loc_temp = targets_loc

                    inputs_loc = torch.zeros((inputs_loc_temp.shape[0], size[0], size[1], size[2] , inputs_loc_temp.shape[4])).to(self.exp.get_from_config('device'))
                    targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], size[2] , targets_loc_temp.shape[4])).to(self.exp.get_from_config('device'))

                    full_res_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res.shape[4])).to(self.exp.get_from_config('device'))
                    full_res_gt_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res.shape[4])).to(self.exp.get_from_config('device'))

                    factor = self.exp.get_from_config('train_model') - m -1
                    factor_pow = math.pow(2, factor)

                    for b in range(inputs_loc.shape[0]): 
                        while True:
                            pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                            pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])
                            pos_z = random.randint(0, inputs_loc_temp.shape[3] - size[2])
                            #print(pos_x)
                            #print(pos_y)
                            #print(pos_z)

                            #if not torch.any(targets_loc_temp):
                            #    break

                            #print(torch.unique(targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]))
                            #if len(torch.unique(targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :])) == 2:
                            
                            #print(torch.sum(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0:1] == 0))
                            #print(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0:1].shape)
                            #print(torch.numel(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0:1]) )
                            if torch.sum(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0:1] == 0) / torch.numel(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0:1]) < 0.3:
                                #if 1 in torch.unique(targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]):
                                break
                                #else:
                                #    print("NO MASK")
                            #else:
                            #    print("SKIP")
                                
                            #else:
                            #    print("NOOOOOO")

                        # SIZE OF FULL RES
                        pos_x_full = int(pos_x * factor_pow)
                        pos_y_full = int(pos_y * factor_pow)
                        pos_z_full = int(pos_z * factor_pow)

                        size_full = [int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac)]


                        
                        #pos_x = random.randint(0, full_res.shape[1] - size[0])
                        #pos_y = random.randint(0, full_res.shape[2] - size[1])
                        #pos_z = random.randint(0, full_res.shape[3] - size[2])

                        #print(inputs_loc_temp.shape)
                        #print(inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :].shape)
                        #print(size[2] )

                        # ----------- SET
                        inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        if len(targets_loc.shape) > 4:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        else:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]
                       

                        full_res_new[b] = full_res[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]
                        full_res_gt_new[b] = full_res_gt[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]

                        #full_res = inputs_loc
                        #full_res_label = targets_loc
                    full_res = full_res_new
                    full_res_gt = full_res_gt_new

                    # NOW CROP


        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)

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