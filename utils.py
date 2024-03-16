import torchvision
from torch.utils.data import DataLoader
import config


def load_data():
    train = torchvision.datasets.MNIST(
        root="./data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    train_data = DataLoader(train, batch_size=config.BATCH_TRAIN, shuffle=True)

    test = torchvision.datasets.MNIST(
        root="./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    test_data = DataLoader(test, batch_size=config.BATCH_TEST, shuffle=True)

    return train_data, test_data


