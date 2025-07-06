from src.utils import minimum_spanning_tree, euclidean_distance
import torch
from torch.nn.functional import normalize
from math import prod

class PersistenceLoss(torch.nn.Module):
    def __init__(self, dist_fn =  euclidean_distance):
        super().__init__()
        self.dist_fn = dist_fn

    def forward(self, X_tensor: torch.Tensor, Z_tensor: torch.Tensor):
        return persistence_loss(X_tensor, Z_tensor, self.dist_fn)

def persistence_loss(X_tensor: torch.Tensor, Z_tensor: torch.Tensor, dist_fn =  euclidean_distance):
    """
        Compute perstence loss between two tensors 
    """
    # Normalize for both to make sure the consistency loss
    X_norm = normalize(X_tensor, dim=0) / prod(s for s in X_tensor.shape[1:])
    Z_norm = normalize(Z_tensor, dim=0) / prod(s for s in Z_tensor.shape[1:])
    
    pi_X = minimum_spanning_tree(X_tensor, dist_fn)
    
    AX_pi = [dist_fn(X_norm[x], X_norm[y])  for  x, y in pi_X]
    AZ_pi = [dist_fn(Z_norm[x], Z_norm[y]) for x, y in pi_X]

    loss = sum((x - y)**2 for x, y in zip(AX_pi, AZ_pi))
    return loss / X_tensor.shape[0] * prod(s for s in X_tensor.shape[1:])     # Scalar tensor with grad