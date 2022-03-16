import torch

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