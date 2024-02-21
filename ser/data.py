from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def prepare_data(batch_size):
    """Prepare training and validation data loaders."""
    # Torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Data loaders
    training_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return training_dataloader, validation_dataloader
