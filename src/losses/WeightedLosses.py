import torch
import torch.nn as nn
import src.losses.DiceBCELoss


class WeightedLosses(nn.Module):
    def __init__(self, config):
        super(WeightedLosses, self).__init__()
        assert len(config['trainer.losses']) == len(config['trainer.loss_weights'])
        self.losses = []
        self.weights = []
        for i, _ in enumerate(config['trainer.losses']):
            self.losses.append(eval(config['trainer.losses'][i])())
            self.weights.append(config['trainer.loss_weights'][i])
            

    def forward(self, pred, target):
        loss = 0
        loss_ret = {}
        for i, _ in enumerate(self.losses):
            l, d = self.losses[i](pred, target)
            loss += l * self.weights[i]
            for k, v in d.items():
                loss_ret[f"{self.losses[i].__class__.__name__}/{k}"] = d[k] * self.weights[i]
        return loss, loss_ret