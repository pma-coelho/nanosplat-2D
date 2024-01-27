import torch

from .renderer import GaussianSplatRenderer


class GaussianSplatSolver:

    def __init__(self, config):
        self.ng = config['n_gaussians']
        self.config = config

        if torch.cuda.is_available(): 
            self.device = "cuda:0" 
        else: 
            self.device = "cpu"
        print(f"Running on {self.device}...")

    def solve(self, image):
        image = image.to(self.device)

        res = image.shape[0], image.shape[1]

        self.renderer = GaussianSplatRenderer(res, self.device)

        positions       = torch.rand(self.ng, 2, device=self.device) * torch.tensor(res, device=self.device)
        angles          = torch.rand(self.ng   , device=self.device) * torch.pi
        scaling_factors = torch.rand(self.ng, 2, device=self.device) * self.config['max_init_scale']
        opacities       = torch.rand(self.ng   , device=self.device) / self.ng
        colors          = torch.rand(self.ng, 3, device=self.device) * 255

        parameters = tuple([positions, angles, scaling_factors, opacities, colors])

        result = self.renderer.draw(*parameters)

        return result