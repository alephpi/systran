import argparse

import torch

from train import get_checkpoints


def main():
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
    num_checkpoints = len(checkpoints)
    print("Averaging %d checkpoints" % num_checkpoints)

    averaged_model = None

    for checkpoint in checkpoints:
        print("Loading %s" % checkpoint)
        checkpoint = torch.load(checkpoint, map_location="cpu")

        model = checkpoint["model"]
        model = {key: value / num_checkpoints for key, value in model.items()}

        if averaged_model is None:
            averaged_model = model
        else:
            for key, value in model.items():
                averaged_model[key] += value

    print("Saving %s" % output_path)
    checkpoint = {"model": averaged_model}
    torch.save(checkpoint, output_path)


if __name__ == "__main__":
    main()
