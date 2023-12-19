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
import pandas as pd
import seaborn as sns
import plotly.express as px

class Agent_NCA_gen(Agent_NCA, Agent_MedSeg3D):
    """Base agent for training NCA models
    """

    def save_state_2(self, model_path: str) -> None:
        r"""Save state of current model
        """
        super().save_state(model_path)
        #if self.pool.__len__() != 0 and self.exp.get_from_config('save_pool'):
        #    dump_compressed_pickle_file(self.pool, os.path.join(model_path, 'pool.pbz2'))
        dump_compressed_pickle_file(self.exp.dataset.data, 'data.pbz2')

    def load_state_2(self, model_path):
        r"""Load state - Add Pool to state
        """
        super().load_state(model_path)
        if os.path.exists(os.path.join(model_path, 'pool.pbz2')):
            self.pool = load_compressed_pickle_file(os.path.join(model_path, 'pool.pbz2'))


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

        data_dict = {'id':id, 'image':inputs, 'label':targets, 'image_vec':vec}
        return data_dict
    
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
        self.optimizer_backpropTrick = optim.SGD(backdrop_trick_params, lr=self.exp.get_from_config('lr')*256)#, betas=self.exp.get_from_config('betas'))
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.exp.get_from_config('lr'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
        self.scheduler_backprop = optim.lr_scheduler.ExponentialLR(self.optimizer_backpropTrick, 0.9995)

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

                vec_loss = loss_f(outputs[i, ...], targets[i, ...])
                v = v.to(self.device) - ((1.0-self.model.list_backpropTrick[i].weight.squeeze()).detach())#*vec_loss
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

    def reset_weights(self):
        # reset weights
        for i in range(self.batch_size):
            new_weight = self.model.list_backpropTrick[i].weight.data.clone()

            # Modify the cloned tensor
            new_weight[new_weight != 1] = 1.0
            self.model.list_backpropTrick[i].weight.data = new_weight         

    def normalize(self):
        #self.exp.set_model_state('train')
        data_loader = torch.utils.data.DataLoader(self.exp.dataset, shuffle=True, batch_size=10000) 
        vec = []
        for data in data_loader:
            vec = data['image_vec']#.detach().cpu().numpy()
            id = data['id']

        normalized = self.basic_normalize(vec)


        for d in range(vec.shape[1]):#s in zip(id, normalized):
            idx = id[d].split('_')[0]
            self.exp.dataset.set_vec(int(idx), normalized[d].detach().cpu().numpy())
    
    def basic_normalize(self, vec):
        vec = vec - torch.min(vec)
        vec = vec / torch.max(vec)
        vec = vec * 2 - 1

        return vec

    def z_score_normalize(self, vec):
        normalized = vec

        for d in range(vec.shape[1]):
            mean_val = sum(vec[:, d]) / len(vec[:, d])
            std_dev = (sum((x - mean_val) ** 2 for x in vec[:, d]) / len(vec[:, d])) ** 0.5
            for pos, x in enumerate(vec[:, d]):
                normalized[pos, d] = (x - mean_val) / std_dev

        return normalized


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
                    self.exp.dataset.set_vec(v)
                    # reset weights
                    self.embedding_backpropTrick.weight[self.embedding_backpropTrick.weight != 1] = 1.0

                # reset model weights
        return loss_ret
    
    def intermediate_evaluation(self, *args) -> None:
        super().intermediate_evaluation(*args)

        self.fitUMAP(args[1])
    
    @torch.no_grad()
    def fitUMAP(self, epoch):
        self.exp.set_model_state('train')
        data_loader = torch.utils.data.DataLoader(self.exp.dataset, shuffle=True, batch_size=10000) 
        vec = []
        for data in data_loader:
            vec = data['image_vec'].detach().cpu().numpy()
            id = data['id']
            labels = data['class']

        #labels = []
        #for i in id:
        #    print("ID: ", i)
        #    idx = i.split('_')[0]
        #    labels.append(int(idx) % 2)

        # ------------------------------- GENEREATE Boxplot
        df = pd.DataFrame(vec)
        df['id'] = id
        df['class'] = labels  
        df_long = df.melt(id_vars=['id', 'class'], var_name='feature', value_name='value')

        # Create a Box Plot
        fig = px.box(df_long, x='feature', y='value', color='class', title='Feature Distribution per Class')
        self.exp.write_figure('Boxplot', fig, epoch)


        # ------------------------------- GENEREATE PAIRPLOT

        if self.model.extra_channels <= 8:
            df = pd.DataFrame(vec)
            df['id'] = id
            df['class'] = labels

            # If the DataFrame has many columns, you might need to rename them for clarity
            df.columns = [f'dim_{i}' if i < df.shape[1] - 2 else df.columns[i] for i in range(df.shape[1])]

            # Visualize with Plotly's scatter matrix
            fig = px.scatter_matrix(df, 
                                    dimensions=[f'dim_{i}' for i in range(vec.shape[1])],
                                    color='class')
            self.exp.write_figure('Pairplot', fig, epoch)



        print("FITUMAP")
        print(vec)
        print("SUM: ", np.sum(vec))
        # ------------------------------- GENEREATE UMAP

        return
        if len(id) > 10:
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(vec)

            df = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])
            df['label'] = labels

            # Plotting with Plotly Express
            fig = px.scatter(df, x='UMAP-1', y='UMAP-2', color='label', title='UMAP projection')

            # Write the figure using the method of your experiment object
            # Ensure this method is capable of handling Plotly figures
            self.exp.write_figure('UMAP', fig, epoch)

        if False:
            if self.model.extra_channels == 1:
                img_plot = plt.scatter(vec, np.zeros(shape = vec.shape), c=labels, s=20, cmap='Set1').get_figure() #, cmap='Spectral'
                #plt.gca().set_aspect('equal', 'datalim')
                #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
                #img_plot = plt.title('UMAP projection of the Digits dataset', fontsize=24).get_figure()
                #img_plot = plt
                self.exp.write_figure('Simple1D plot', img_plot, epoch)
            elif self.model.extra_channels == 2:
                img_plot = plt.scatter(vec[:, 0], vec[:, 1], c=labels, s=20, cmap='Set1').get_figure() #, cmap='Spectral'
                #plt.gca().set_aspect('equal', 'datalim')
                #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
                #img_plot = plt.title('UMAP projection of the Digits dataset', fontsize=24).get_figure()
                #img_plot = plt
                self.exp.write_figure('Simple2D plot', img_plot, epoch)

        return
    
    def plot_results_byPatient(self, loss_log: dict):
        r"""Plot losses in a per patient fashion with seaborn to display in tensorboard.
            #Args
                loss_log ({name: loss}: Dictionary of losses
        """
        loss_data = list(loss_log.items())
        df = pd.DataFrame(loss_data, columns=['Epoch', 'Loss'])

        # Create a scatter plot using Plotly Express
        fig = px.scatter(df, x='Epoch', y='Loss', title='Loss Over Epochs')

        # Set the range of the y-axis
        fig.update_yaxes(range=[0, 1])

        # If you need to return the figure (e.g., for further processing or saving)
        return fig
        print(loss_log)
        sns.set_theme()
        plot = sns.scatterplot(x=loss_log.keys(), y=loss_log.values())
        plot.set(ylim=(0, 1))
        plot = plot.get_figure()
        return plot