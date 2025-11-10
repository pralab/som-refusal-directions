import numpy as np
import torch 

def compute_centroid(X, layer):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    return X[:, layer, :].mean(dim=0)
