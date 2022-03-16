import os
import torch
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
from torch.utils.tensorboard import SummaryWriter


class Experiment():
    def __init__(self, config, dataset, model):
        self.projectConfig = config
        self.config = self.projectConfig[0]
        self.dataset = dataset
        self.model = model

        if(os.path.isdir(self.config['model_path'])):
            self.reload()
        else:
            self.setup()

        self.general()
        print(config)
        return
    
    def setup(self):
        # Create dirs
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'models'), exist_ok=True)
        # Create basic configuration
        self.data_split = DataSplit(self.config['img_path'], self.config['label_path'], data_split = self.config['data_split'], dataset = self.dataset)
        dump_pickle_file(self.data_split, os.path.join(self.config['model_path'], 'data_split.dt'))
        dump_json_file(self.projectConfig, os.path.join(self.config['model_path'], 'config.dt'))
        self.currentStep = 0

    def get_max_steps(self):
        return self.projectConfig[-1]['n_epoch']

    def reload(self):
        # TODO: Proper reload
        self.currentStep = self.current_step()
        self.data_split = load_pickle_file(os.path.join(self.config['model_path'], 'data_split.dt'))
        self.projectConfig = load_json_file(os.path.join(self.config['model_path'], 'config.dt'))
        self.config = self.projectConfig[0]
        self.reload_model()
        print("Reload State " + str(self.currentStep))
        #self.setup()
        
    def general(self):
        self.dataset.set_size(self.config['input_size'])
        self.writer = SummaryWriter(log_dir=self.get_from_config('model_path'))
        self.set_current_config()

    def reload_model(self):
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep), 'model.pth')
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self):
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep+1))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        #self.model.load_state_dict(torch.load(model_path))

    def current_step(self):
        dirs = [d for d in os.listdir(os.path.join(self.config['model_path'], 'models')) if os.path.isdir(os.path.join(os.path.join(self.config['model_path'], 'models'), d))]
        if dirs:
            maxDir = max([int(d.split('_')[1]) for d in dirs])
            return maxDir
        else: 
            return 0

    def set_model_state(self, state):
        self.dataset.setPaths(self.config['img_path'], self.data_split.get_images(state), self.config['label_path'], self.data_split.get_labels(state))
        
    def get_from_config(self, tag):
        return self.config[tag]
        #for i in reversed(range(len(self.projectConfig))):
        #    if self.projectConfig[i]['n_epoch'] > self.currentStep:
        #        if tag in self.projectConfig[i]:
        #            return self.projectConfig[i][tag]
        #return None

    def set_current_config(self):
        self.config = {}
        for i in range(0, len(self.projectConfig)):
            for k in self.projectConfig[i].keys():
                self.config[k] = self.projectConfig[i][k]
            if self.projectConfig[i]['n_epoch'] > self.currentStep:
                return

    def increase_step(self):
        self.currentStep = self.currentStep +1
        self.set_current_config()

    def get_current_config(self):
        return self.config

    def write_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def write_img(self, tag, image, step):
        self.writer.add_image(tag, image, step, dataformats='HWC')


class DataSplit():
    def __init__(self, path_image, path_label, data_split, dataset):
        self.images = self.split_files(self.getFilesInFolder(path_image, dataset), data_split)
        self.labels = self.split_files(self.getFilesInFolder(path_label, dataset), data_split)

    def get_images(self, state):
        return self.get_data(self.images[state])

    def get_labels(self, state):
        return self.get_data(self.labels[state])

    def get_data(self, data):
        lst = data.values()
        lst_out = []
        for d in lst:
            lst_out.extend([*d.values()])
        return lst_out

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

    def getFilesInFolder(self, path, dataset):
        return  dataset.getFilesInPath(path) #os.path.listdir(path) # length [f for f in os.path.listdir(path) if os.path.isfile(os.path.join(path, f))]
    