from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.model import Net
from ser.train import train as train_model, validate_model
from ser.data import prepare_data

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs for training."
    ),
    batch_size: int = typer.Option(
        ..., "-b", "--batch", help="Batch size for training."
    ),
    learning_rate: float = typer.Option(
      0.01, "-l", "--learning", "Learning rate for training."  
    )
):
    print(f"Running experiment: {name}\n"
          f"Epochs: {epochs}\n"
          f"Batch Size: {batch_size}\n"
          f"Learning Rate: {learning_rate}\n")
          
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # dataloaders
    training_dataloader, validation_dataloader = prepare_data(batch_size)

    # train
    train_model(model, optimizer, training_dataloader, validation_dataloader, epochs, device)

    validate_model(model, validation_dataloader, epochs, device)


@main.command()
def infer():
    print("This is where the inference code will go")