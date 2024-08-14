import torch


class OverflowLoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        super().__init__()
    

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        loss_ret = {}
        loss = 0

        hidden: torch.Tensor = kwargs['hidden_channels']

        hidden_overflow_loss = (hidden - torch.clip(hidden, -1.0, 1.0)).abs().mean()
        with torch.no_grad():
            _max = torch.amax(target, dim=tuple(range(1,len(target.shape))), keepdim=True)
            _min = torch.amin(target, dim=tuple(range(1,len(target.shape))), keepdim=True)


        rgb_overflow_loss = (output - torch.clamp(output, _min, _max)).abs().mean()

        loss = hidden_overflow_loss + rgb_overflow_loss
        loss_ret['hidden'] = hidden_overflow_loss.item()
        loss_ret['rgb'] = rgb_overflow_loss.item()

        return loss, loss_ret