import torch
import numpy as np
from src.utils.helper import load_compressed_pickle_file
from src.agents.Agent_NCA import Agent
import os
import random
import torchio as tio
import math
import nibabel as nib

class Agent_M3D_NCA(Agent):
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
        for m in range(self.exp.get_from_config('train_model')+1):
            self.optimizer[m].zero_grad()
        loss = 0
        loss_ret = {}
        for m in range(outputs.shape[-1]):
            if 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            for m in range(self.exp.get_from_config('train_model')+1):
                self.optimizer[m].step() 
                self.scheduler[m].step()
        return loss_ret

    def get_outputs(self, data, full_img=False, tag="", **kwargs):
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
        avg_pool = torch.nn.MaxPool3d(2, 2, 0)
        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        

        targets_loc = targets 

        # Scale Image to Initial Size
        full_res = inputs
        full_res_gt = targets
        inputs_loc = inputs

        for i in range(self.exp.get_from_config('train_model')*int(math.log2(scale_fac))):
            inputs_loc = inputs_loc.transpose(1,4)
            inputs_loc = avg_pool(inputs_loc)
            inputs_loc = inputs_loc.transpose(1,4)
            targets_loc = targets_loc.transpose(1,4)
            targets_loc = max_pool(targets_loc)
            targets_loc = targets_loc.transpose(1,4)

        input_channel = self.exp.get_from_config('input_channels')

        outputs_img = []
        outputs_targets = []
        

        if full_img == True:

            # REMOVE: Just for shortterm visualisation
            save4d = False
            slice_all_Channels = False
            if not slice_all_Channels:
                label_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1], inputs.shape[2], inputs.shape[3]))
                img_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            else:
                x_size = math.ceil(math.sqrt(inputs.shape[4]))
                label_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1]*x_size, inputs.shape[2]*x_size,1), dtype=float)
                img_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1]*x_size, inputs.shape[2]*x_size,1), dtype=float)
            step = 0
            # -------------------------
            
            with torch.no_grad():
                for m in range(self.exp.get_from_config('train_model')+1):
                    
                    if m == self.exp.get_from_config('train_model'):
                        if type(self.getInferenceSteps()) is list:
                            stp = self.getInferenceSteps()[m]
                        else:
                            stp = self.getInferenceSteps()
                        if save4d:
                            outputs = inputs_loc
                            for i in range(self.getInferenceSteps()[m]):
                                outputs = self.model[m](outputs, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                if not slice_all_Channels:
                                    label_mri_4d[step, ...] = outputs[0, ..., 1].detach().cpu().numpy() 
                                    img_mri_4d[step, ...] = inputs_loc[0, ..., 0].detach().cpu().numpy() 
                                else:
                                    for x in range(4):
                                        for y in range(4):
                                            label_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = outputs[0, ..., 11, x+y*4].detach().cpu().numpy() 
                                            img_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = inputs_loc[0, ..., 11, x+y*4].detach().cpu().numpy() 

                                step = step +1 
                        else:
                            outputs = self.model[m](inputs_loc, steps=stp, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                    else:
                        up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')

                        if save4d:
                            outputs = inputs_loc
                            for i in range(self.getInferenceSteps()[m]):
                                outputs = self.model[m](outputs, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                if not slice_all_Channels:
                                    label_mri_4d[step, ...] = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 1].detach().cpu().numpy() 
                                    img_mri_4d[step, ...] = torch.permute(up(torch.permute(inputs_loc, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 0].detach().cpu().numpy() 
                                else:
                                    for x in range(4):
                                        for y in range(4):
                                            label_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 11, x+y*4].detach().cpu().numpy() 
                                            img_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = torch.permute(up(torch.permute(inputs_loc, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 11, x+y*4].detach().cpu().numpy()                                     
                                step = step +1 


                        else:
                            outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        

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

                        print(next_res.shape, outputs.shape)
                        inputs_loc = torch.concat((next_res[...,:input_channel], outputs[...,input_channel:]), 4)
                        #print(inputs_loc.shape)
                        targets_loc = targets

            if save4d:
                if not slice_all_Channels:
                    nib_save = torch.sigmoid(torch.from_numpy(np.transpose(label_mri_4d, (1, 2, 3, 0)))).numpy()
                    nib_save[nib_save>0.5] = 1
                    nib_save[nib_save != 1] = 0 
                else:
                    nib_save = torch.from_numpy(np.transpose(label_mri_4d, (1, 2, 3, 0))).numpy() 
                    sign = nib_save<0
                    
                nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) 
                nib.save(nib_save, os.path.join("/home/jkalkhof_locale/Documents/temp/Test4D/", str(id)+"_"+tag+".nii.gz"))

                nib_save = torch.sigmoid(torch.from_numpy(np.transpose(img_mri_4d, (1, 2, 3, 0)))).numpy()
                nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                nib.save(nib_save, os.path.join("/home/jkalkhof_locale/Documents/temp/Test4D/", str(id)+"_img.nii.gz"))
        else:

            for m in range(self.exp.get_from_config('train_model')+1): 

                if m == self.exp.get_from_config('train_model'):
                    if type(self.getInferenceSteps()) is list:
                        stp = self.getInferenceSteps()[m]
                    else:
                        stp = self.getInferenceSteps()
                    outputs = self.model[m](inputs_loc, steps=stp, fire_rate=self.exp.get_from_config('cell_fire_rate'))
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
   
                    inputs_loc = torch.concat((next_res[...,:input_channel], outputs[...,input_channel:]), 4)

                    
                    targets_loc = next_res_gt

                    size = self.exp.get_from_config('input_size')[0]

                    inputs_loc_temp = inputs_loc
                    targets_loc_temp = targets_loc

                    inputs_loc = torch.zeros((inputs_loc_temp.shape[0], size[0], size[1], size[2] , inputs_loc_temp.shape[4])).to(self.exp.get_from_config('device'))
                    targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], size[2] , targets_loc_temp.shape[4])).to(self.exp.get_from_config('device'))

                    full_res_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res.shape[4])).to(self.exp.get_from_config('device'))
                    full_res_gt_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res_gt.shape[4])).to(self.exp.get_from_config('device'))

                    factor = self.exp.get_from_config('train_model') - m -1
                    factor_pow = math.pow(2, factor)

                    for b in range(inputs_loc.shape[0]): 
                        while True:
                            pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                            pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])
                            pos_z = random.randint(0, inputs_loc_temp.shape[3] - size[2])
                            break

                        # SIZE OF FULL RES
                        pos_x_full = int(pos_x * factor_pow)
                        pos_y_full = int(pos_y * factor_pow)
                        pos_z_full = int(pos_z * factor_pow)

                        size_full = [int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac)]

                        # ----------- SET
                        inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        if len(targets_loc.shape) > 4:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        else:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

                        full_res_new[b] = full_res[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]
                        full_res_gt_new[b] = full_res_gt[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]

                    full_res = full_res_new
                    full_res_gt = full_res_gt_new

                    # NOW CROP


        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc 

    def resize4d(self, img, size=(64,64), factor=4, label=False):
        if label:
            transform = tio.Resize((size[0], size[1], size[2], -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], size[2], -1))
        img = transform(img)
        return img

    def random_crop(self, img, label, outputs):
        size = self.exp.get_from_config('input_size')[0]
        pos_x = random.randint(0, img.shape[1] - size[0])
        pos_y = random.randint(0, img.shape[2] - size[1])

        transform = tio.Resize((int(img.shape[1]), int(img.shape[2]), -1))
        outputs = transform(outputs.cpu()) 
        outputs = outputs.to(self.exp.get_from_config('device'))
        outputs = outputs[:, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]

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