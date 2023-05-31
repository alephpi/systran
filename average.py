import argparse

import torch

from train import get_checkpoints
from utils import init_logger, get_logger


def main():
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", help="Directory containing the checkpoints")
    parser.add_argument(
        "--output",
        default="averaged_checkpoint.pt",
        help="Path to the averaged checkpoint",
    )
    parser.add_argument(
        "--max_to_average",
        type=int,
        default=10,
        help="Maximum number of checkpoints to average",
    )
    args = parser.parse_args()

    checkpoints = get_checkpoints(args.checkpoints_dir)
    checkpoints = checkpoints[-args.max_to_average :]

    average_checkpoints(checkpoints, args.output)


def average_checkpoints(checkpoints, output_path):
    logger = get_logger()

    num_checkpoints = len(checkpoints)
    logger.info("Averaging %d checkpoints", num_checkpoints)

    averaged_model = None

    for checkpoint in checkpoints:
        logger.info("Loading %s", checkpoint)
        checkpoint = torch.load(checkpoint, map_location="cpu")

        model = checkpoint["model"]
        model = {key: value / num_checkpoints for key, value in model.items()}

        if averaged_model is None:
            averaged_model = model
        else:
            for key, value in model.items():
                averaged_model[key] += value

    logger.info("Saving %s", output_path)
    checkpoint = {"model": averaged_model}
    torch.save(checkpoint, output_path)


if __name__ == "__main__":
    main()
