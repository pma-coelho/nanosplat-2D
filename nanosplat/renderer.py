import torch
import numpy as np


def angle_to_2d_rot_matrix(angle):
    """Convert an angle (radians) into a rotation matrix where the first column vector and the x-axis make the specified angle."""

    sin = torch.sin(angle)
    cos = torch.cos(angle)
    return torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1)
    ], dim=-1)


def gaussian2D(x, position, angle, scaling_factors):
    """
    Compute the density of an unnormalized 2D gaussian distribution at x.
    
    To avoid matrix inversions and redundant computations the covariance matrix is not explicitly computed.
    
    Instead, we decompose the squared mahalanobis distance term inside the exponential,
    (x-p) @ S^(-1) @ (x-p).T , into (x-p) @ R.T @ s^(-1) @ R @ (x-p).T,
    where R is an orthogonal rotation matrix calculated from the angle,
    and s is the diagonal matrix of scaling factors.

    We then compute the squared mahalanobis distance as D @ D.T,
    where D = R @ (x-p).T @ s^(-1/2)
    """

    rot_matrix = angle_to_2d_rot_matrix(angle)
    deltas = x[..., None, :] - position
    rotated_deltas = torch.sum(rot_matrix * deltas[..., None, :], dim=-1)
    scaled_deltas = rotated_deltas / torch.sqrt(scaling_factors)
    return torch.exp(-0.5 * torch.sum(scaled_deltas**2, dim=-1))


class GaussianSplatRenderer:

    def __init__(self, res):
        
        # Keep canvas in memory to avoid new memory allocations
        self.canvas = torch.zeros(res + (3,))

        # Precompute pixel indexes
        self.x = torch.from_numpy(np.stack(np.mgrid[:res[0], :res[1]], axis=-1))
        self.x.requires_grad = False


    def draw(self, positions, angles, scaling_factors, opacities, colors):
        
        densities = gaussian2D(self.x, positions, angles, scaling_factors)
        splats = densities[..., None] * opacities[..., None] * colors
        self.canvas = torch.sum(splats, dim=-2)

        return self.canvas