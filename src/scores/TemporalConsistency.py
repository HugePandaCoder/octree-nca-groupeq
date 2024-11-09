# ChatGPT :)
import torch, einops

def mean_temporal_flicker(predictions):
    """
    Compute the Mean Temporal Flicker for a batch of predicted video segmentations.
    
    Parameters:
    - predictions (torch.Tensor): Predicted segmentation tensor of shape (B, C, H, W, T).
    
    Returns:
    - float: Mean temporal flicker across all frames and batch.
    """
    # Calculate the absolute difference between consecutive frames along the temporal dimension (T)
    flicker = torch.abs(predictions[..., 1:] - predictions[..., :-1])
    return flicker.mean()

def temporal_consistency_error(predictions, ground_truth):
    """
    Compute the Temporal Consistency Error (TCE) for a batch of predicted video segmentations.
    
    Parameters:
    - predictions (torch.Tensor): Predicted segmentation tensor of shape (B, C, H, W, T).
    - ground_truth (torch.Tensor): Ground truth segmentation tensor of shape (B, C, H, W, T).
    
    Returns:
    - float: Temporal Consistency Error across all frames and batch.
    """
    # Compute per-pixel class matches between predictions and ground_truth
    matches = (predictions == ground_truth).float()  # 1 if classes match, 0 otherwise
    
    # Calculate the difference in matches between consecutive frames (1 indicates change, 0 indicates consistency)
    temporal_changes = torch.abs(matches[..., 1:] - matches[..., :-1])
    
    # Sum changes across the spatial dimensions (H, W) and the class dimension (C)
    changes_sum = temporal_changes.sum(dim=(1, 2, 3, 4))  # Summed across B, C, H, W for each time diff
    
    # Mean TCE across batch and time
    mean_tce = changes_sum.mean() / (predictions.size(1) * predictions.size(2) * predictions.size(3))
    
    return mean_tce

# Example usage
# predictions = torch.rand(B, C, H, W, T) > 0.5  # Example binary prediction tensor (randomized for illustration)
# ground_truth = torch.rand(B, C, H, W, T) > 0.5  # Example binary ground truth tensor (randomized for illustration)

# flicker = mean_temporal_flicker(predictions)
# tce = temporal_consistency_error(predictions, ground_truth)

# print("Mean Temporal Flicker:", flicker)
# print("Temporal Consistency Error (TCE):", tce)

class MeanTemporalFlickerScore(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs):
        pred = pred > 0.0
        pred = pred.float()
        pred = einops.rearrange(pred, 'b h w t c -> b c h w t')
        
        res = mean_temporal_flicker(pred)

        pred = einops.rearrange(pred, 'b c h w t -> b h w t c')
        return res
    

class TemporalConsistencyScore(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs):
        pred = pred > 0.0
        pred = pred.float()
        pred = einops.rearrange(pred, 'b h w t c -> b c h w t')
        
        target = einops.rearrange(target, 'b h w t c -> b c h w t')
        
        res = temporal_consistency_error(pred, target)

        pred = einops.rearrange(pred, 'b c h w t -> b h w t c')
        target = einops.rearrange(target, 'b c h w t -> b h w t c')
        return res
