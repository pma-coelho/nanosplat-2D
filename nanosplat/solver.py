import torch

from .renderer import GaussianSplatRenderer


class GaussianSplatSolver:

    def __init__(self, config):
        self.ng = config['n_gaussians']
        self.config = config
        

    def solve(self, image):
        res = image.shape[0], image.shape[1]

        self.renderer = GaussianSplatRenderer(res)

        positions       = torch.rand(self.ng, 2) * torch.tensor(res)
        angles          = torch.rand(self.ng) * torch.pi
        scaling_factors = torch.rand(self.ng, 2) * self.config['max_init_scale']
        opacities       = torch.rand(self.ng) / self.ng
        colors          = torch.rand(self.ng, 3) * 255
                                              
        result = self.renderer.draw(positions, angles, scaling_factors, opacities, colors)

        return result