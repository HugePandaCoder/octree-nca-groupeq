import torch
import numpy as np
from src.utils.helper import dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent_NCA import Agent_NCA
import os
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D
import umap
import matplotlib.pyplot as plt
import hdbscan
import torch.optim as optim

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
        id, inputs, targets, vec = data['id'], data['image'], data['label'], data['image_vec']
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
        id, inputs, targets, vec = data['id'], data['image'], data['label'], data['image_vec']
        outputs = self.model(inputs, vec, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        #outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

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
        backdrop_trick_params = list(self.model.embedding_backpropTrick.parameters())

        # Filter out the backdrop trick parameters from all parameters
        #filtered_params = [param for param in all_params if not any((param.data == p.data).all() for p in backdrop_trick_params)]
        #filtered_params = [param for param in all_params if param not in backdrop_trick_params]


        # Create an optimizer for the filtered parameters
        self.optimizer = optim.Adam(all_params, lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.optimizer_backpropTrick = optim.SGD(backdrop_trick_params, lr=self.exp.get_from_config('lr')*1000)#, betas=self.exp.get_from_config('betas'))
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.exp.get_from_config('lr'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

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
            # Individual vector loss

            # FOR element in batch
            
            standard = True
            if standard:
                self.optimizer_backpropTrick.zero_grad()
                loss.backward()
                self.optimizer_backpropTrick.step()

            for i, v in enumerate(vec):
            # Add optimizer trick here
                
                # BACKWARD LOSS VEC   
                if not standard:
                    self.optimizer_backpropTrick.zero_grad()
                    vec_loss = loss_f(outputs[i, ...], targets[i, ...])
                    vec_loss.backward(retain_graph=True)
                    self.optimizer_backpropTrick.step()
                
                
                # MOVE loss to vector
                v_id = str(id[i])
                v_id = int(v_id.split('_')[0])
                v = v.to(self.device) - (1.0-self.model.embedding_backpropTrick.weight.squeeze()).detach()*10
                #v = torch.clip(v, -1, 1)
                self.exp.dataset.set_vec(v_id, v.detach().cpu().numpy())
                if not standard:
                    self.reset_weights()  
            # reset weights
            if standard:
                self.reset_weights() 
            else:      
                loss.backward()
            
            
            self.optimizer.step()
            self.scheduler.step()
            self.reset_weights() 
            self.z_score_normalize()


        return loss_ret

    def reset_weights(self):
        # reset weights
        new_weight = self.model.embedding_backpropTrick.weight.data.clone()

        # Modify the cloned tensor
        new_weight[new_weight != 1] = 1.0
        self.model.embedding_backpropTrick.weight.data = new_weight         
                

    def z_score_normalize(self):
        #self.exp.set_model_state('train')
        data_loader = torch.utils.data.DataLoader(self.exp.dataset, shuffle=True, batch_size=10000) 
        vec = []
        for data in data_loader:
            vec = data['image_vec'].detach().cpu().numpy()
            id = data['id']

        normalized = vec

        for d in range(vec.shape[1]):
            mean_val = sum(vec[:, d]) / len(vec[:, d])
            std_dev = (sum((x - mean_val) ** 2 for x in vec[:, d]) / len(vec[:, d])) ** 0.5
            for pos, x in enumerate(vec[:, d]):
                normalized[pos, d] = (x - mean_val) / std_dev

        for s in zip(id, normalized):
            idx = s[0].split('_')[0]
            self.exp.dataset.set_vec(int(idx), s[1])
        



    def batch_step2(self, data: tuple, loss_f: torch.nn.Module) -> dict:
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
    
    def intermediate_evaluation(self, *args) -> None:
        super().intermediate_evaluation(*args)

        self.fitUMAP(args[1])
    
    def fitUMAP(self, epoch):
        self.exp.set_model_state('train')
        data_loader = torch.utils.data.DataLoader(self.exp.dataset, shuffle=True, batch_size=10000) 
        vec = []
        for data in data_loader:
            vec = data['image_vec'].detach().cpu().numpy()
            id = data['id']

        labels = []
        for i in id:
            print("ID: ", i)
            idx = i.split('_')[0]
            labels.append(int(idx) % 2)

        print("FITUMAP")
        print(vec)
        print("SUM: ", np.sum(vec))

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(vec)

        img_plot = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, cmap='bwr').get_figure() #, cmap='Spectral'
        #plt.gca().set_aspect('equal', 'datalim')
        #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        #img_plot = plt.title('UMAP projection of the Digits dataset', fontsize=24).get_figure()
        #img_plot = plt
        self.exp.write_figure('UMAP', img_plot, epoch)
        #plt.show()

        if self.model.extra_channels == 1:
            img_plot = plt.scatter(vec, np.zeros(shape = vec.shape), c=labels, s=5, cmap='bwr').get_figure() #, cmap='Spectral'
            #plt.gca().set_aspect('equal', 'datalim')
            #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
            #img_plot = plt.title('UMAP projection of the Digits dataset', fontsize=24).get_figure()
            #img_plot = plt
            self.exp.write_figure('Simple1D plot', img_plot, epoch)
        elif self.model.extra_channels == 2:
            img_plot = plt.scatter(vec[:, 0], vec[:, 1], c=labels, s=5, cmap='bwr').get_figure() #, cmap='Spectral'
            #plt.gca().set_aspect('equal', 'datalim')
            #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
            #img_plot = plt.title('UMAP projection of the Digits dataset', fontsize=24).get_figure()
            #img_plot = plt
            self.exp.write_figure('Simple1D plot', img_plot, epoch)

        return