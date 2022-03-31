import os
import torch
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
from torch.utils.tensorboard import SummaryWriter

r"""This class handles:
        - Interactions with the experiment folder
        - Loading / Saving experiments
        - Datasets
"""
class Experiment():
    def __init__(self, config, dataset, model, agent):
        self.projectConfig = config
        self.config = self.projectConfig[0]
        self.dataset = dataset
        self.model = model
        self.agent = agent
        self.general()
        if(os.path.isdir(os.path.join(self.config['model_path'], 'models'))):
            self.reload()
        else:
            self.setup()
        self.currentStep = self.currentStep+1

    
    r"""Initial experiment setup when first started
    """
    def setup(self):
        # Create dirs
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))), exist_ok=True)
        # Create basic configuration
        self.data_split = DataSplit(self.config['img_path'], self.config['label_path'], data_split = self.config['data_split'], dataset = self.dataset)
        dump_pickle_file(self.data_split, os.path.join(self.config['model_path'], 'data_split.dt'))
        dump_json_file(self.projectConfig, os.path.join(self.config['model_path'], 'config.dt'))

    r"""This function is useful for evaluation purposes where you want to change the config, e.g. data paths or similar.
        It does not save the config and should NEVER be used during training.
    """
    def temporarly_overwrite_config(self, config):
        print("WARNING: NEVER USE \'temporarly_overwrite_config\' FUNCTION DURING TRAINING.")
        self.projectConfig = config
        self.set_current_config()

    r"""Get max defined training steps of experiment
    """
    def get_max_steps(self):
        return self.projectConfig[-1]['n_epoch']

    r"""Reload old experiment to continue training
        TODO: Add functionality to load any previous saved step
    """
    def reload(self):
        # TODO: Proper reload
        self.data_split = load_pickle_file(os.path.join(self.config['model_path'], 'data_split.dt'))
        self.projectConfig = load_json_file(os.path.join(self.config['model_path'], 'config.dt'))
        self.config = self.projectConfig[0]
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep))
        if os.path.exists(model_path):
            print("Reload State " + str(self.currentStep))
            self.agent.load_state(model_path)
        #self.setup()
        
    r"""General experiment configurations needed after setup or loading
    """
    def general(self):
        self.currentStep = self.current_step()
        self.dataset.set_size(self.config['input_size'])
        self.writer = SummaryWriter(log_dir=os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))))
        self.set_current_config()
        self.agent.set_exp(self)
        if self.currentStep == 0:
            self.write_text('config', str(self.projectConfig), 0)

    r"""Reload model
        TODO: Move to a more logical position. Probably to the model and then call directly from the agent
    """
    def reload_model(self):
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep), 'model.pth')
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            #self.model.load_state_dict(torch.load(model_path))

    r"""TODO: Same as for reload -> move to better location
    """
    def save_model(self):
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep+1))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        #self.model.load_state_dict(torch.load(model_path))

    r"""Find out the initial epoch by checking the saved models"""
    def current_step(self):
        model_path = os.path.join(self.config['model_path'], 'models')
        if os.path.exists(model_path):
            dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(os.path.join(self.config['model_path'], 'models'), d))]
            if dirs:
                maxDir = max([int(d.split('_')[1]) for d in dirs])
                return maxDir
        return 0

    r"""TODO: remove? """
    def set_model_state(self, state):
        self.dataset.setPaths(self.config['img_path'], self.data_split.get_images(state), self.config['label_path'], self.data_split.get_labels(state))
        
    r"""Get from config
        Args:
            tag (String): Key of requested value
    """
    def get_from_config(self, tag):
        return self.config[tag]

    r"""Set current config. This can change during training and will always 
        overwrite previous settings, but keep everything else
    """
    def set_current_config(self):
        self.config = {}
        for i in range(0, len(self.projectConfig)):
            for k in self.projectConfig[i].keys():
                self.config[k] = self.projectConfig[i][k]
            if self.projectConfig[i]['n_epoch'] > self.currentStep:
                return

    r"""Increase current epoch
    """
    def increase_epoch(self):
        self.currentStep = self.currentStep +1
        self.set_current_config()

    r"""TODO: remove?"""
    def get_current_config(self):
        return self.config

    r"""Write scalars to tensorboard
    """
    def write_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    r"""Write an image to tensorboard
    """
    def write_img(self, tag, image, step):
        self.writer.add_image(tag, image, step, dataformats='HWC')

    r"""Write text to tensorboard
    """
    def write_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)

    r"""Write data as histogram to tensorboard
    """
    def write_histogram(self, tag, data, step):
        self.writer.add_histogram(tag, data, step)


r"""Handles the splitting of data
"""
class DataSplit():
    def __init__(self, path_image, path_label, data_split, dataset):
        self.images = self.split_files(self.getFilesInFolder(path_image, dataset), data_split)
        self.labels = self.split_files(self.getFilesInFolder(path_label, dataset), data_split)

    r"""Returns the images of selected state
        Args:
            state (String): Can be 'train', 'val', 'test'
    """
    def get_images(self, state):
        return self.get_data(self.images[state])

    r"""Returns the labels of selected state
        Args:
            state (String): Can be 'train', 'val', 'test'
    """
    def get_labels(self, state):
        return self.get_data(self.labels[state])

    r"""Returns the data in a list rather than the stored folder strucure
        Args:
            data ({}): Dictionary ordered by {id, {slice, img_name}}
    """
    def get_data(self, data):
        lst = data.values()
        lst_out = []
        for d in lst:
            lst_out.extend([*d.values()])
        return lst_out

    r"""Split files into train, val, test according to definition
        while keeping patients slics together.
        Args:
            files ({int, {int, string}}): {id, {slice, img_name}}
            data_split ([float, float, float]): Sum of 1
    """
    def split_files(self, files, data_split):
        dic = {'train':{}, 'val':{}, 'test':{}}
        for index, key in enumerate(files):
            if index / len(files) < data_split[0]:
                dic['train'][key] = files[key]
            elif index / len(files) < data_split[0] + data_split[1]: 
                dic['val'][key] = files[key]
            else:
                dic['test'][key] = files[key]
        return dic

    r"""Get files in folder
        Args:
            path (String): Path to folder
            dataset (Dataset)
    """
    def getFilesInFolder(self, path, dataset):
        return  dataset.getFilesInPath(path) #os.path.listdir(path) # length [f for f in os.path.listdir(path) if os.path.isfile(os.path.join(path, f))]
    