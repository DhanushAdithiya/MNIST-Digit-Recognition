from model import CNN
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import load_data, tensor_to_img 
from PIL import Image
import config


train, test = load_data()


model = CNN(config.IN_CHANNELS, config.OUT_CHANNELS)
model.cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.NLLLoss()

epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in train:
        # Flatten MNIST images into a 784 long vector

        # Training pass
        optimizer.zero_grad()

        output = model(images.cuda())
        loss = criterion(output, labels.cuda())

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train)))


#testing model performance
with torch.no_grad():
    correct = 0
    total = 0

    for image, label in test:
        for i in range(len(label)):
            if total == 10:
                break

            img = image[i]
            outputs = model(image.cuda())

            ps = torch.exp(outputs)
            prob = list(ps.cpu().numpy()[0])
            pred = prob.index(max(prob))
            true = label.numpy()[i]
            if (pred == true):
                correct += 1
            total += 1

            print(f"Correct: {true} - Predicted : {pred}")
            save_image(img, f"./generated/{total}.png")


    accuracy = (correct / total) * 100
    print(accuracy)

