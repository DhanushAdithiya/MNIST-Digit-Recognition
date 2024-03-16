from model import CNN
import torch
import torch.nn as nn
from utils import load_data
import config

def main():
    train, test = load_data()

    model = CNN()
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.9)



    for _, (image,label) in enumerate(train):
        image, label = image.to(config.DEVICE), label.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()




    correct = 0
    t_loss = 0
    total = 0

    with torch.no_grad():
        for image, label in test:
            image, label = image.to(config.DEVICE), label.to(config.DEVICE)
            output = model(image)
            t_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += 1

    accuracy = correct / total
    print(f"accuracy: {accuracy}%") 



if __name__ == "__main__":
    main()
