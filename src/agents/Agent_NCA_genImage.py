import torch
import numpy as np
from src.utils.helper import dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent_NCA import Agent_NCA
import os
from src.agents.Agent_NCA_gen import Agent_NCA_gen
from src.agents.Agent_Growing import Agent_Growing
import umap
import matplotlib.pyplot as plt
import hdbscan
import torch.optim as optim
import pandas as pd
import seaborn as sns
import plotly.express as px
import torch.nn.functional as F

class Agent_NCA_genImage(Agent_NCA_gen):
    """Base agent for training NCA models
    """

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        id, inputs, targets, vec = data['id'], data['image'], data['label'], data['image_vec']
        outputs, targets = self.get_outputs(data)
        #plt.imshow(targets[0, ...].detach().cpu().numpy())
        #plt.axis('off')  # Optional: Turn off axis labels and ticks
        #plt.show()
        #plt.imshow(outputs[0, ...].detach().cpu().numpy())
        #plt.axis('off')  # Optional: Turn off axis labels and ticks
        #plt.show()
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        #if len(outputs.shape) == 4:
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

        loss = F.mse_loss(outputs[...], targets[...]) #F.l1_loss(outputs[...], targets[...]) + 
        loss_ret[0] = loss.item()

        if loss != 0:
            # Individual vector loss

            # FOR element in batch
            
            standard = True
            if standard:
                self.optimizer_backpropTrick.zero_grad()
                loss.backward()
                self.optimizer_backpropTrick.step()


            print(self.model.list_backpropTrick[0].weight.squeeze().detach())
            for i, v in enumerate(vec):
            # Add optimizer trick here
                
                # BACKWARD LOSS VEC   
                if not standard:
                    self.optimizer_backpropTrick.zero_grad()
                    vec_loss = loss_f(outputs[i, ...], targets[i, ...])
                    vec_loss.backward(retain_graph=True)
                    self.optimizer_backpropTrick.step()
                    print("NOT STANDARD")
                
                
                # MOVE loss to vector
                v_id = str(id[i])
                v_id = int(v_id.split('_')[0])

                
                #vec_loss = F.mse_loss(outputs[i, ...], targets[i, ...])
                
                v = v.to(self.device) - ((1.0-self.model.list_backpropTrick[i].weight.squeeze()).detach())*24000#*10000000#*vec_loss
                #print(v)
                v = torch.clip(v, -1, 1)
                self.exp.dataset.set_vec(v_id, v.detach().cpu().numpy())
                if not standard:
                    self.reset_weights()  
            # reset weights
            if standard:
                self.reset_weights() 
            else:      
                loss.backward()
            
            if False:
                if self.exp.currentStep % 20 < 10:
                    self.optimizer.step()
                    self.scheduler.step()
                if int(self.exp.currentStep % 10) == 0 and int(self.exp.currentStep % 20) != 0:
                    #self.reset_vecs()
                    for i in id:
                        idx = i.split('_')[0]
                        if True:
                            if i.__contains__('hippocampus'):
                                img_vec = np.array([0.5, 0.05, 0.05]).astype(np.float32)
                            elif i.__contains__('prostate'):
                                img_vec = np.array([0.05, 0.5, 0.05]).astype(np.float32)
                            elif i.__contains__('liver'):
                                img_vec = np.array([0.05, 0.05, 0.5]).astype(np.float32)
                        self.exp.dataset.set_vec(int(idx), img_vec.astype(np.float32))
            self.optimizer.step()
            self.scheduler.step()
            self.scheduler_backprop.step()
            self.reset_weights() 
            #self.normalize()


        return loss_ret

    def initialize(self):
        r"""Initialize agent with optimizers and schedulers
        """
        super().initialize()
        self.device = torch.device(self.exp.get_from_config('device'))
        self.batch_size = self.exp.get_from_config('batch_size')
        # If stacked NCAs

        # Get all parameters from the model
        all_params = list(self.model.parameters())

        # Get parameters from the backdrop trick
        #backdrop_trick_params = list(self.model.embedding_backpropTrick.parameters())
        backdrop_trick_params = []
        for i in range(self.batch_size):
            backdrop_trick_params.extend(self.model.list_backpropTrick[i].parameters())

        # Filter out the backdrop trick parameters from all parameters
        #filtered_params = [param for param in all_params if not any((param.data == p.data).all() for p in backdrop_trick_params)]
        #filtered_params = [param for param in all_params if param not in backdrop_trick_params]


        # Create an optimizer for the filtered parameters
        self.optimizer = optim.Adam(all_params, lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        #self.optimizer_backpropTrick = optim.Adam(backdrop_trick_params, lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas')) #optim.SGD(backdrop_trick_params, lr=16e-4)#, betas=self.exp.get_from_config('betas'))
        self.optimizer_backpropTrick = optim.SGD(backdrop_trick_params, lr=16e-1)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.exp.get_from_config('lr'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
        self.scheduler_backprop = optim.lr_scheduler.ExponentialLR(self.optimizer_backpropTrick, 0.99995)

    @torch.no_grad()
    def test(self, *args, **kwargs):
        r"""Test the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        # Prepare dataset for testing
        dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # For each data sample
        #for i, data in enumerate(dataloader):
        #    data = self.prepare_data(data, eval=True)
        #    data_id, inputs, _ = data['id'], data['image'], data['label']
        #    outputs, targets = self.get_outputs(data, full_img=True, tag="0")

        #outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate')).detach().cpu().numpy()


        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            outputs, targets = self.get_outputs(data, full_img=True, tag="0")
            #if i == 0:
            #    print(outputs.shape)
            #    plt.imshow(outputs[0, ...].detach().cpu().numpy())
            #    plt.axis('off')  # Optional: Turn off axis labels and ticks
            #    plt.show()
            self.exp.write_img(str(data_id), (outputs[0, ...].detach().cpu().numpy()), self.exp.currentStep)