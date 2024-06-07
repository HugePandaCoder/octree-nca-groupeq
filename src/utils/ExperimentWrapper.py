import torch.utils.data
from src.agents.Agent import BaseAgent
from src.utils.Experiment import Experiment
from torch.utils.data import RandomSampler

class ExperimentWrapper():
    def createExperiment(self, config : dict, model, agent: BaseAgent, dataset, loss_function):
        model.to(config['device'])
        exp = Experiment(config, dataset, model, agent)
        
        if config['batchgenerators']:
            data_loader = iter(dataset)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'),
                                                    num_workers=exp.get_from_config('num_workers'))

        exp.set_loss_function(loss_function)
        exp.set_data_loader(data_loader)
        return exp