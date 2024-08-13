import torch
import torch.nn as nn
import src.losses.DiceBCELoss
import src.losses.OverflowLoss


class WeightedLosses(nn.Module):
    def __init__(self, config):
        super(WeightedLosses, self).__init__()
        assert len(config['trainer.losses']) == len(config['trainer.loss_weights']), f"{config['trainer.losses']} and {config['trainer.loss_weights']} must have the same length"
        self.losses = []
        self.weights = []
        for i, _ in enumerate(config['trainer.losses']):
            self.losses.append(eval(config['trainer.losses'][i])())
            self.weights.append(config['trainer.loss_weights'][i])
            

    def forward(self, pred, target, **kwargs):
        loss = 0
        loss_ret = {}
        for i, _ in enumerate(self.losses):
            try:
                r = self.losses[i](pred, target, **kwargs)
            except TypeError:
                r = self.losses[i](pred, target)
            if isinstance(r, tuple):
                l, d = r
            else:
                l = r
                d = {'loss': l.item()}
            loss += l * self.weights[i]
            for k, v in d.items():
                loss_ret[f"{self.losses[i].__class__.__name__}/{k}"] = d[k] * self.weights[i]
        return loss, loss_ret