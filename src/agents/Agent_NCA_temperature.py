import torch
import numpy as np
from src.utils.helper import convert_image, dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from IPython.display import clear_output
from src.utils.helper import dump_pickle_file, load_pickle_file
import os
from src.agents.Agent_NCA import Agent
import torch.nn as nn
import torch.nn.functional as F
import math

class Agent_Temperature(Agent):
    
    def set_temperature(self, valid_loader, model):
            """
            Tune the tempearature of the model (using the validation set).
            We're going to set it to optimize NLL.
            valid_loader (DataLoader): validation set loader
            """
            #self.cuda()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = _ECELoss().cuda()

            # First: collect all the logits and labels for the validation set
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for id, input, label in valid_loader:
                    id, input, label = self.prepare_data((id, input, label))
                    #input = input.cuda()
                    #logits = self.model(input)
                    logits, label = self.get_outputs((id, input, label))
                    label = label.type(torch.LongTensor) 
                    for x in range(logits.shape[1]):
                        for y in range(logits.shape[2]):
                            logits_list.append(logits[:, x, y, 0:2])
                            labels_list.append(label[:, x, y, 1])
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()

            print(logits.shape)
            print(labels.shape)
            #logits = torch.unsqueeze(logits, 1)
            #logits = torch.cat((logits, logits), 0)
            #labels = torch.unsqueeze(labels, 1)

            # Calculate NLL and ECE before temperature scaling
            #logits = logits[0]

            before_temperature_nll = nll_criterion(logits, labels).item()
            before_temperature_ece = ece_criterion(logits, labels).item()
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50) #self.temperature

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(model.temperature_scale_2(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(model.temperature_scale_2(logits), labels).item()
            after_temperature_ece = ece_criterion(model.temperature_scale_2(logits), labels).item()
            print('Optimal temperature: %.3f' % model.temperature.item())
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

            return self

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1) #dim=1
        print(softmaxes.shape)
        confidences, predictions = torch.max(softmaxes, 1)
        print(confidences.shape)
        print(predictions.shape)
        print(labels.shape)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece