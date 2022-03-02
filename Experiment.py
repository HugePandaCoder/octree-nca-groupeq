import os

class Experiment():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        if(config['reload'] == True):
            self.reload()
        else:
            self.setup()

        print(config)
        return
    
    def setup(self):
        self.data_split = DataSplit(self.config['img_path'], self.config['label_path'], data_split = self.config['data_split'], dataset = self.dataset)

    def reload(self):
        # TODO: Proper reload
        self.setup()
        return

    def set_model_state(self, state):
        self.dataset.setPaths(self.config['img_path'], self.data_split.get_images(state), self.config['label_path'], self.data_split.get_labels(state))
        

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
    