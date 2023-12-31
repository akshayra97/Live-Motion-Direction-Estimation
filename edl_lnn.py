# -*- coding: utf-8 -*-
"""EDL- LNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c6iGx_bP0y-r8dmsa-BOUf_AXe0PVFiD
"""

import numpy as np
import os
from os.path import join, isfile
from PIL import Image
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# Change directory to the data directory from hw1
os.chdir("/content/drive/MyDrive/EDL/Disparity_Map/Disparity_Map")
half_path = "/content/drive/MyDrive/EDL/Disparity_Map/Disparity_Map"

count = 0
train_x_im = []
dir_list = os.listdir('.')
train_labels = []
label_count = 0
# Open all the images from the files list for both train and test data and append to a large tensor
# Normalize the data so it is easier to process
for folder in dir_list:
    full_path = half_path + "/" + folder
    os.chdir(full_path)
    for file in os.listdir('.'):
        with Image.open(file) as img:
            img = img.resize((300, 300))
            img_data = np.asarray(img)
            img_data = img_data.astype(np.float32) # / 255.0
            img_tensor = torch.from_numpy(img_data).float()
            train_x_im.append(img_tensor)
        train_labels.append(label_count)
    label_count = label_count + 1
train_x = torch.stack(train_x_im)
print(np.shape(train_x))
print(np.shape(train_labels))
# Rearange tensor
#train_shuf, labels_shuf = shuffle(train_x, train_labels, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_labels, test_size=0.33, random_state=42)
X_train = X_train.unsqueeze(0)
X_test = X_test.unsqueeze(0)
X_train = X_train.permute(1, 0, 2, 3)
X_test = X_test.permute(1, 0, 2, 3)
print(np.shape(X_train))
# Create a tensor for the labels as well
y_train = np.array(y_train)
y_train = y_train.astype(int)
y_test = np.array(y_test)
y_test = y_test.astype(int)
train_label = torch.tensor(y_train).long()
test_label = torch.tensor(y_test).long()

import torch.nn as nn
# Create CNN model with various layers. Apply pooling and RELU
# Make sure to set the linear layers correctly
# class Conv2D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
            
#             nn.Conv2d(1,20,5),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(20,64,5),
#             nn.ReLU(),
#             nn.MaxPool2d(4,4),
#             nn.Conv2d(64,64,7),
#             nn.ReLU(),
#             nn.MaxPool2d(4,4),
            
#             nn.Flatten(),
#             #nn.Linear(3136,6000),
#             #nn.ReLU(),
#             nn.Linear(3136,1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512,36)
#         )
    
#     def forward(self, x):
#         return self.network(x)

# model = Conv2D()

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(90000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
    
    def forward(self, x):
        return self.network(x)

model = LinearModel()

# Create your train and test datasets with labels
batch_size = 200
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train,train_label),
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test,test_label),
                                           batch_size=batch_size,
                                           shuffle=True)
 
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
train_loss_tot = []
train_acc_tot = []
test_loss_tot = []
test_acc_tot = []
 
# Loop through the number of epochs
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
 
    # set model to train mode
    model.train()
    # iterate over the training data
    for inputs, labels in train_loader:
        #inputs = inputs.reshape(batch_size,1,300,300)
        optimizer.zero_grad()
        outputs = model(inputs)
        #compute the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # increment the running loss and accuracy
        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()
 
    # calculate the average training loss and accuracy
    train_loss /= len(train_loader)
    train_loss_tot.append(train_loss)
    train_acc /= len(train_loader.dataset)
    train_acc_tot.append(train_acc)
 
    # set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            #inputs = inputs.reshape(batch_size,1,300,300) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).sum().item()
 
    # calculate the average validation loss and accuracy
    test_loss /= len(test_loader)
    test_loss_tot.append(test_loss)
    test_acc /= len(test_loader.dataset)
    test_acc_tot.append(test_acc)
    print(f'Epoch {epoch+1}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')

# Plot the training and validation loss
plt.plot(train_loss_tot, label='train loss')
plt.plot(test_loss_tot, label='test loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(train_acc_tot, label='train acc')
plt.plot(test_acc_tot, label='test acc')
plt.legend()
plt.show()

