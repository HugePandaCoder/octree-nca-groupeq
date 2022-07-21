import torch
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


class DiceFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
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