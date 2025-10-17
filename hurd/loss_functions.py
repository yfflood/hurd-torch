import torch
import numpy as np


def mse(predictions, targets):
    """Mean squared error loss function"""
    # Get device from predictions
    device = predictions.device if isinstance(predictions, torch.Tensor) else torch.device('cpu')
    
    # Ensure both are torch tensors on the same device
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.from_numpy(predictions).float().to(device)
    if not isinstance(targets, torch.Tensor):
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(device)
        else:
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
    elif targets.device != device:
        targets = targets.to(device)
    
    return torch.mean((predictions[:, 1] - targets) ** 2)


def crossentropy(predictions, targets, eps=1e-30):
    """Crossentropy loss function"""
    # Get device from predictions
    device = predictions.device if isinstance(predictions, torch.Tensor) else torch.device('cpu')
    
    # Ensure both are torch tensors on the same device
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.from_numpy(predictions).float().to(device)
    if not isinstance(targets, torch.Tensor):
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(device)
        else:
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
    elif targets.device != device:
        targets = targets.to(device)
    
    # deal with targets being single probabilities.
    # targets are p_choose_B by CPC convention so
    # p_choose_A is 1 - targets
    if len(targets.shape) != 2:
        targets = torch.vstack([1 - targets, targets]).T
    predictions = predictions + eps
    ce_per_datapoint = -torch.sum(targets * torch.log(predictions), axis=1)
    
    return torch.mean(ce_per_datapoint)
