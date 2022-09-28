import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import math
#import torchmetrics as tm

# TODO: License -> https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        print(target.unique())
        intersection = (input * target).sum()
        print(2.*intersection)
        print(input.sum())
        print(target.sum())
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)
        print(dice)

        return 1 - dice

class DiceLoss_mask(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input, target, mask = None, smooth=1):
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        mask = torch.flatten(mask)

        input = input[~mask]  
        target = target[~mask]  
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceLossV2(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        super(DiceLossV2, self).__init__()

    def forward(self, input, target, smooth=1):
        if self.useSigmoid:
            input = torch.sigmoid(input)  

        return 1 - tm.functional.dice(input, target)



class DiceBCELoss_Distance(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        self.gradientScaling = None
        super(DiceBCELoss_Distance, self).__init__()

    def gradient(self, size):
        img_gr = np.zeros((size[1], size[2]))

        a = np.array((0,0))
        b = np.array((size[1]/2, size[2]/2))        
        max_distance = np.linalg.norm(a - b)

        for x in range(size[1]):
            for y in range(size[2]):
                a = np.array((x, y))
                img_gr[x, y] =  np.linalg.norm(a - b) / max_distance#1 - math.sqrt(math.pow(((size[1]/2) - x) / (size[1]/2) + ((size[2]/2) - y) / (size[2]/2), 2)) /2
        #plt.imshow(img_gr)
        #plt.show()

        return img_gr

    def forward(self, input, target, smooth=1):

        if self.gradientScaling is None:
            self.gradientScaling = self.gradient(input.shape)

        img_gr = np.stack([self.gradientScaling] * input.shape[0], axis=0)  
        img_gr = torch.Tensor(img_gr).to(input.get_device())

        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        distance = torch.flatten(img_gr)
        
        intersection = (input * target * distance).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.mean((torch.nn.functional.binary_cross_entropy(input, target, reduction='none') * distance))
        Dice_BCE = BCE + dice_loss
        Dice_BCE = dice_loss

        return Dice_BCE


class DiceBCELoss(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    def __init__(self, useSigmoid = True):
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

# MIT License
#
# Copyright (c) 2017 Ke Ding
# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        loss = loss_bce * (1 - logit) ** self.gamma  # focal loss
        loss = loss.mean()
        return loss


class DiceFocalLoss(FocalLoss):

    def __init__(self, gamma=2, eps=1e-7):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal