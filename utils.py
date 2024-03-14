import torchvision
from torch.utils.data import DataLoader
import config
import numpy as np
import matplotlib.pyplot as plt


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


def tensor_to_img(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    image = image.numpy().astype(np.uint8)
    return image


def view_classify(img, ps):
    """
    Function for saving an image and its predicted classes to a file.
    """
    ps = ps.cpu().data.numpy().squeeze()

    # Image processing (same as before)
    img = img.resize_(1, 28, 28).numpy().squeeze()

    # Generate filename based on class with highest probability (optional)
    class_id = np.argmax(ps)  # Get index of class with highest probability
    filename = f"./img/image_{class_id}.png"  # Example filename

    # Save the image
    plt.imsave(filename, img, cmap="gray" if len(img.shape) == 2 else None)

