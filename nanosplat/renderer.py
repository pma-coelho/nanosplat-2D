import torch

from itertools import product

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    
    To avoid matrix inversions and redundant computations, the covariance matrix is not explicitly computed.
    
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

    def __init__(self, res, device, config):

        self.res = res
        self.device = device
        self.config = config

        # Precompute pixel indices
        self.x = torch.tensor(np.stack(np.mgrid[:res[0], :res[1]], axis=-1)).to(device)

        # Precompute tile splices
        self.tile_slices = self.get_tile_slices()


    def get_tile_slices(self):
        n = self.config['num_tiles_per_side']
        tile_bounds = np.stack([
            np.arange(0, self.res[0] + 1, self.res[0] / n).astype(int),
            np.arange(0, self.res[1] + 1, self.res[1] / n).astype(int)
        ])
        tile_bounds = sliding_window_view(tile_bounds, (2, 2)).squeeze()
        tile_slices = [
            (
                slice(bounds_0[0].item(), bounds_0[1].item()),
                slice(bounds_1[0].item(), bounds_1[1].item())
            )
            for bounds_0, bounds_1 in product(tile_bounds[:, 0], tile_bounds[:, 1])
        ]
        return tile_slices


    def get_splat_cull_mask(self, tile_slice, positions, scaling_factors):

        min_0, max_0 = tile_slice[0].start, tile_slice[0].stop
        min_1, max_1 = tile_slice[1].start, tile_slice[1].stop

        tolerance = self.config['tolerance_factor'] * torch.max(torch.abs(scaling_factors), dim=-1, keepdim=True)[0]

        min_cull = positions + tolerance > torch.tensor([min_0, min_1]).to(self.device)
        max_cull = positions - tolerance < torch.tensor([max_0, max_1]).to(self.device)

        return torch.logical_and(
            torch.prod(min_cull, dim=-1),
            torch.prod(max_cull, dim=-1)
        )

    def draw(self, positions, angles, scaling_factors, colors):
        canvas = torch.zeros(self.res + (3,), device=self.device)
        for tile in self.tile_slices:
            keep = self.get_splat_cull_mask(tile, positions, scaling_factors)
            densities = gaussian2D(self.x[tile], positions[keep], angles[keep], scaling_factors[keep])
            splats = densities[..., None] * colors[keep]
            canvas[tile] = torch.sum(splats, dim=-2)
        return canvas