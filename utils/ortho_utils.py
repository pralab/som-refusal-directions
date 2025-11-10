import os
import torch 

def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)
    proj = torch.einsum('...d,d->...', matrix, vec)  # shape: [...]
    return matrix - proj.unsqueeze(-1) * vec  # shape: [..., d_model]

def orthogonalize_weights(model, direction: torch.Tensor):
    model.orthogonalize_weights(direction)
    