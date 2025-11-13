import os
import torch 

def get_ablated_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)
    proj = torch.einsum('...d,d->...', matrix, vec)  # shape: [...]
    return matrix - proj.unsqueeze(-1) * vec  # shape: [..., d_model]

def ablate_weights(model, direction: torch.Tensor):
    model.ablate_weights(direction)
    