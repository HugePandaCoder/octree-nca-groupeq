import torch


class BCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid: bool = True, smooth: float = 1) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super().__init__()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            BCE = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='mean')
        else:
            BCE = torch.nn.functional.binary_cross_entropy(output, target, reduction='mean')
        
        return BCE
    

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss_ret = {}
        loss = 0
        if len(output.shape) == 5 and target.shape[-1] == 1:
            for m in range(target.shape[-1]):
                loss_loc = self.compute(output[..., m], target[...])
                loss = loss + loss_loc
                loss_ret[f"mask_{m}"] = loss_loc.item()
        else:
            for m in range(target.shape[-1]):
                if 1 in target[..., m]:
                    loss_loc = self.compute(output[..., m], target[..., m])
                    loss = loss + loss_loc
                    loss_ret[f"mask_{m}"] = loss_loc.item()

        return loss, loss_ret