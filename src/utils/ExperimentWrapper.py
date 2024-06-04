import torch.utils.data
from src.agents.Agent import BaseAgent
from src.utils.Experiment import Experiment
from torch.utils.data import RandomSampler

class ExperimentWrapper():
    def createExperiment(self, config : dict, model, agent: BaseAgent, dataset, loss_function):
        model.to(config['device'])
        exp = Experiment(config, dataset, model, agent)

        sampler = None
        if exp.get_from_config('num_steps_per_epoch') is not None:
            sampler = RandomSampler(dataset, replacement=True, num_samples=exp.get_from_config('num_steps_per_epoch') * exp.get_from_config('batch_size'))
        
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=sampler is None, batch_size=exp.get_from_config('batch_size'),
                                                num_workers=exp.get_from_config('num_workers'), sampler=sampler)
        

        exp.set_loss_function(loss_function)
        exp.set_data_loader(data_loader)
        return exp