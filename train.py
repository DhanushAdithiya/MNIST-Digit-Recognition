from model import CNN
import torch
import torch.nn as nn
from utils  import load_data
import config


train, test = load_data()

model = CNN(config.IN_CHANNELS, config.OUT_CHANNELS)
model.cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.NLLLoss()

epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train)))
