import torch
import os

from tqdm import tqdm

from .renderer import GaussianSplatRenderer
from .utils import save_image

class GaussianSplatSolver:

    def __init__(self, config):
        self.ng = config["n_gaussians"]
        self.config = config

        # Select compute device
        if torch.cuda.is_available(): 
            self.device = "cuda:0" 
        else: 
            self.device = "cpu"
        print(f"Running on {self.device}...")


    def init_parameters(self):
        '''
        Initialize gaussian splat parameters uniformly over their ranges.

        :return: Parameters to be optimized
        :rtype: tuple[torch.tensor]
        '''

        scaling_factor_range = self.config["max_init_scale"] - self.config["min_init_scale"] 

        positions       = torch.rand(self.ng, 2) * torch.tensor(self.res)
        angles          = torch.rand(self.ng) * torch.pi
        scaling_factors = torch.rand(self.ng, 2) * scaling_factor_range + self.config["min_init_scale"] 
        colors          = torch.rand(self.ng, 3) * self.config["max_init_color"]

        # Move parameters to the compute device and configure them as leafs of the compute graph
        parameters = list()
        for p in positions, angles, scaling_factors, colors:
            device_p = p.to(self.device).detach()
            device_p.requires_grad = True
            parameters.append(device_p) 

        return tuple(parameters)


    @staticmethod
    def mse_loss(x, y):
        '''L2 pixel loss for to image tensors of equal shape'''

        return torch.mean((x - y)**2)


    def solve(self, image, output_folder=None):
        '''
        Create gaussian splat approximation of input image.

        :param torch.tensor image: Image tensor with shape (3, H, W)
        :param str output_folder: The recipient of the message

        :return: Gaussian splat image tensor with shape (3, H, W)
        :rtype: torch.tensor
        '''

        image = image.to(self.device)
        self.res = image.shape[0], image.shape[1]

        self.parameters = self.init_parameters()
        self.renderer = GaussianSplatRenderer(self.res, self.device, self.config["renderer"])

        optimizer = torch.optim.Adam(self.parameters, lr=self.config["init_lr"])

        iter_bar = tqdm(range(self.config["max_iters"]))
        for i in iter_bar:

            # Render gaussian splats and compute loss
            result = self.renderer.draw(*self.parameters)
            loss = self.mse_loss(image, result)

            # Update loss in progress bar
            iter_bar.set_postfix(loss=loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save checkpoints
            if output_folder != None and i % 100 == 0:
                save_image(result.detach().to("cpu"), os.path.join(output_folder, "checkpoints", f"iter_{i}.png"))

        iter_bar.close()
        return result.detach().to("cpu")