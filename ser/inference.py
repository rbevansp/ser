# inference.py

import torch
from pathlib import Path
from ser.data import test_dataloader
from ser.transforms import transforms, normalize
from ser.params import Params
import json

def run_infer(run_path: str):
    label = 6

    # Load saved parameters
    params_file = Path(run_path) / "params.json"
    with open(params_file, "r") as f:
        params_data = json.load(f)
    params = Params(**params_data)

    # Print experiment summary
    print("Experiment Name:", params.name)
    print("Hyperparameters:")
    print("\tEpochs:", params.epochs)
    print("\tBatch Size:", params.batch_size)
    print("\tLearning Rate:", params.learning_rate)

    # Select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # Load the model
    model = torch.load(Path(run_path) / "model.pt")

    # Run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    pixels = images[0][0]

    # Generate ASCII art representation
    ascii_art = generate_ascii_art(pixels)

    print(f"Predicted Label: {pred} | Prediction Confidence: {confidence:.2%}")


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = [pixel_to_char(pixel) for pixel in row]
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    char_map = {
        1.0: "O",
        0.9: "o",
        0.1: ".",
        0.0: " ",
    }
    return char_map.get(pixel, " ")