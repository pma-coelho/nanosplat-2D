import argparse
import os
import shutil

from nanosplat import GaussianSplatSolver
from nanosplat.utils import load_json, load_image, save_image


# Example usage:

# python convert_image.py examples/lena.png
# python convert_image.py examples/golden_gate.png --config=path/to/custom/config.json
# python convert_image.py examples/golden_gate.png --results_folder=path/to/custom/results_folder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image.")
    parser.add_argument("--config_path", help="Path to JSON configuration file.", default="config/default_config.json")
    parser.add_argument("--results_folder", help="Path to JSON configuration file.", default="results")
    args = parser.parse_args()

    # Load config and image
    config = load_json(args.config_path)
    image = load_image(args.image_path, resize=config["resize"])

    # Create output folder and copy config for reproducibility
    image_name = os.path.splitext(os.path.split(args.image_path)[-1])[0]
    output_folder = os.path.join(args.results_folder, image_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shutil.copy(args.config_path, output_folder)

    # Solve
    solver = GaussianSplatSolver(config)
    result = solver.solve(image, output_folder)

    # Save result
    save_image(result, os.path.join(output_folder, "result.png"))