import torchvision
from torch.utils.data import DataLoader
import config
import numpy as np
from PIL import Image


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


def tesnsor_to_img(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5)) + np.array((0.5, 0.5))
    image = image * 255
    image = image.astype(np.uint8)
    return image


def save_image(arr, fp):
    image = Image.fromarray(arr)
    image.save(fp)


