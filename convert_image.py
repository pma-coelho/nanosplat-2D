import argparse
import os

from nanosplat import GaussianSplatSolver

from utils import load_json, load_image, save_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image.")
    parser.add_argument("--config_path", help="Path to JSON configuration file.", default="config/default_config.json")
    parser.add_argument("--output_folder", help="Path to JSON configuration file.", default="results")
    args = parser.parse_args()

    config = load_json(args.config_path)
    image = load_image(args.image_path, resize=config['resize'])

    solver = GaussianSplatSolver(config)
    result = solver.solve(image)

    # Save result
    image_name = os.path.splitext(os.path.split(args.image_path)[-1])[0]
    save_image(result, os.path.join(args.output_folder, image_name, "result.png"))