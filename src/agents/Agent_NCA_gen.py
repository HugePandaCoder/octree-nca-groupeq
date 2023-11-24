import torch
import numpy as np
from src.utils.helper import dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent_NCA import Agent_NCA
import os
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D

class Agent_NCA_gen(Agent_NCA, Agent_MedSeg3D):
    """Base agent for training NCA models
    """

    def prepare_data(self, data, eval=False):
        r"""Prepare the data to be used with the model
            #Args
                data (int, tensor, tensor): identity, image, target mask
            #Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets, vec = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            if self.exp.get_from_config('Persistence'):
                inputs = self.pool.getFromPool(inputs, id, self.device)
            inputs, targets = self.repeatBatch(inputs, targets, self.exp.get_from_config('batch_duplication'))
        return id, inputs, targets, vec
    
    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets, vec = data
        outputs = self.model(inputs, vec, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        #outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets
    
    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        id, inputs, targets, vec = data
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        if len(outputs.shape) == 5:
            for m in range(targets.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[...])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(targets.shape[-1]):
                if 1 in targets[..., m]:
                    loss_loc = loss_f(outputs[..., m], targets[..., m])
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()

            if gradient_norm:
                max_norm = 1.0
                # Gradient normalization
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)


            self.optimizer.step()
            self.scheduler.step()

            # FOR element in batch
            
            for i, v in enumerate(vec):
            # Add optimizer trick here
                #weights = self.model.embedding_backpropTrick.weight
                # shift weights into vec
                #print("VEC", v)
                #print("WEIGHTS", self.model.embedding_backpropTrick.weight.squeeze())
                #print("COMP", v.to(self.device) - (1.0-self.model.embedding_backpropTrick.weight.squeeze()))

                #print("IDDDDD", id[i])
                #"test".split('t')
                v_id = str(id[i])
                v_id = int(v_id.split('_')[0])
                #print("VIDDDDDDD", v_id)
                v = v.to(self.device) - (1.0-self.model.embedding_backpropTrick.weight.squeeze())
                #if i == 0:
                #    print(1.0-self.model.embedding_backpropTrick.weight.squeeze())
                #print("VVVVVVVVVVVVVVv", v)
                self.exp.dataset.set_vec(v_id, v)
                # reset weights
                new_weight = self.model.embedding_backpropTrick.weight.clone()

                # Modify the cloned tensor
                new_weight[new_weight != 1] = 1.0
                self.model.embedding_backpropTrick.weight.data = new_weight
                #self.model.embedding_backpropTrick.weight[self.model.embedding_backpropTrick.weight != 1] = 1.0
                #self.model.embedding_backpropTrick.weight[self.model.embedding_backpropTrick.weight != 1] = 1.0
        return loss_ret

    def batch_step2(self, data: tuple, loss_f: torch.nn.Module) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        id, inputs, targets, vec = data
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

                # FOR element in batch
                for v in vec:
                # Add optimizer trick here
                    weights = self.model[m].embedding_backpropTrick.weight
                    # shift weights into vec
                    v = v - (1.0-self.model[m].embedding_backpropTrick.weight)
                    v_id = id[v].split["_"][0]
                    print("VIDDDDDDD", v_id)
                    self.exp.dataset.set_vec(v)
                    # reset weights
                    self.embedding_backpropTrick.weight[self.embedding_backpropTrick.weight != 1] = 1.0

                # reset model weights
        return loss_ret