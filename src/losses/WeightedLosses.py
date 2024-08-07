import torch
import torch.nn as nn


class WeightedLosses(nn.Module):
    def __init__(self, config):
        super(WeightedLosses, self).__init__()
        assert len(config['trainer.losses']) == len(config['trainer.loss_weights'])
        self.losses = []
        self.weights = []
        for i, _ in enumerate(config['trainer.losses']):
            self.losses.append(eval(config['trainer.losses'][i])())
            self.weights.append(config['trainer.loss_weights'][i])
            

    def forward(self, pred, target) -> torch.Tensor:
        loss = 0
        for i, _ in enumerate(self.losses):
            loss = loss + self.losses[i](pred, target) * self.weights[i]
        return loss