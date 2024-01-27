import torch
import os

from tqdm import tqdm

from .renderer import GaussianSplatRenderer
from .utils import save_image

class GaussianSplatSolver:

    def __init__(self, config):
        self.ng = config["n_gaussians"]
        self.config = config

        if torch.cuda.is_available(): 
            self.device = "cuda:0" 
        else: 
            self.device = "cpu"
        print(f"Running on {self.device}...")


    def init_parameters(self):
        scaling_factor_range = self.config["max_init_scale"] - self.config["min_init_scale"] 

        positions       = torch.rand(self.ng, 2) * torch.tensor(self.res)
        angles          = torch.rand(self.ng) * torch.pi
        scaling_factors = torch.rand(self.ng, 2) * scaling_factor_range + self.config["min_init_scale"] 
        colors          = torch.rand(self.ng, 3) * self.config["max_init_color"]

        parameters = list()
        for p in positions, angles, scaling_factors, colors:
            device_p = p.to(self.device).detach()
            device_p.requires_grad = True
            parameters.append(device_p) 
        return tuple(parameters)
    

    def split_parameters(self):
        positions, angles, scaling_factors, colors = self.parameters

        parameters = torch.repeat_interleave(positions, 3, dim=1)



    @staticmethod
    def mse_loss(x, y):
        return torch.mean((x - y)**2)


    def solve(self, image, output_folder=None):
        image = image.to(self.device)
        self.res = image.shape[0], image.shape[1]

        self.parameters = self.init_parameters()
        self.renderer = GaussianSplatRenderer(self.res, self.device, self.config['renderer'])

        optimizer = torch.optim.Adam(self.parameters, lr=self.config["init_lr"])

        iter_bar = tqdm(range(self.config["max_iters"]))
        for i in iter_bar:

            result = self.renderer.draw(*self.parameters)
            loss = self.mse_loss(image, result)
            iter_bar.set_postfix(loss=loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if output_folder != None and i % 100 == 0:
                save_image(result, os.path.join(output_folder, "checkpoints", f"iter_{i}.png"))

        iter_bar.close()
        return result.detach()