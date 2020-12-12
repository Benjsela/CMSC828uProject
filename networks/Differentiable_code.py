#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Sept 20 16:43:08 2018
For: Starter_kit_cnns_pytorch
Author: Gaurav_Shrivastava 

"""


# Imports

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm 

import sys
sys.path.insert(1, '../saliency')
from network_saliency import visualize_helper_selftrained 


# In[2]:


# 2D Convolution Neural Network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In[3]:


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 784)
        self.fc3 = nn.Linear(784, 784)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x#F.log_softmax(x, dim=1)


# In[4]:


# Batch Training of model
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        noise_sd = torch.randn_like(data, device=device) * 0.25
        data = data + noise_sd
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# In[5]:


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# In[6]:


# def main():
    # Training settings
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = .01
momentum = 0.5
seed = 1
log_interval = 10

torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# for inputs,target in train_loader:
#     break
# # inputs = enumerate(next(train_loader))
# print(torch.randn_like(inputs).shape)#* 2*torch.ones(1,28,28) )
# for epoch in range(1, epochs + 1):
#     train(model, device, train_loader, optimizer, epoch, log_interval)
#     test(model, device, test_loader)

# torch.save(model,'trained_model.pth')


# In[7]:


def soft_argmax(x):
    beta = 2
    # x = torch.Tensor(np.array([[.2, .0, .81, .53, .8]]))
    a = torch.exp(beta*x)
    b = torch.sum(torch.exp(beta*x))
#     print(a,b)
    softmax = a/b
    max = torch.sum(softmax*x,1)
#     print(max)
    pos = x.size()
    
    softargmax = torch.sum(softmax*torch.arange(0,pos[1]))
    return softargmax
#     print(pos, softargmax)
#     mx = softargmax.int()
#     ans = softargmax.round()
# #     print(mx)
#     if ans>mx:
#         return mx +1.0
#     return mx + 0.0

#     print(softargmax.int())#,softmax*torch.arange(0,pos[1]))
    


# In[8]:


class Smooth(object):
#     """A smoothed classifier g """
#     def __init__(self, base_classifier: torch.nn.Module, sigma, epsilon = 0.2):
#         """
#         :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
#         :param num_classes:
#         :param sigma: the noise level hyperparameter
#         :param epsilon: hyperparameter for level of error
#         """
#         self.base_classifier = base_classifier
#         self.sigma = sigma.view(1,28,28)
#         self.target = None
#         self.epsilon = epsilon
        
    """A smoothed classifier g """
    def __init__(self, base_classifier: torch.nn.Module, epsilon = 0.2):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param epsilon: hyperparameter for level of error
        """
        self.base_classifier = base_classifier
        self.sigma = None
        self.target = None
        self.epsilon = epsilon

    def certify(self, x: torch.tensor, sigma, target, n: int, batch_size: int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use for estimation
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
        """
        self.sigma = sigma.view(1,28,28)
        self.target = target +0.0
#         print(target.dtype)
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        cAHat, pABar = self.sample_noise(x,n,batch_size)
#         print(self.sigma)
        if pABar > 0.95:
            pABar = 0.95
        if pABar <0.5:
            radius = self.sigma* 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
#         print(pABar,norm.ppf(pABar))#,radius)
        return cAHat, radius


    def sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        x = x.view(1,28,28)
#         with torch.no_grad():
        counts = 0
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=device) * self.sigma
            scores = self.base_classifier(batch + noise).detach()#.argmax(1)
            predictions = []
            for i in range(len(scores)):
                arg_score =  soft_argmax(scores[i].view(1,-1))
                if torch.abs(arg_score - self.target)<self.epsilon:
                    counts +=1
                predictions.append(arg_score)
#             counts += self._count_arr(predictions, self.num_classes)
            return torch.stack(predictions).mean(), counts/len(scores)


# In[48]:


# Batch Training of model
def train_sigma_network(sigma_model, certified_model, device, train_loader, optimizer, epoch, log_interval):
    sigma_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target + 0.0
        optimizer.zero_grad()
        
        sigma = torch.zeros([len(data),28,28])
        
        # Find the saliency map and pass that into the network
        for i in range(len(data)):
            temp_data = data[i][np.newaxis, ...]
            gradients, max_gradients = visualize_helper_selftrained(model, tensor=temp_data, k=target[i])
            max_gradients = max_gradients[np.newaxis, ...]
            sigma[i] = torch.abs(sigma_model(max_gradients).view(data[i].shape))
        
        total_loss = 0
        pred_loss = 0
        R_loss = 0
        for i in range(len(data)):
            pred_score, R = certified_model.certify(data[i],sigma[i],target[i],100,1)
#             print(R.max(),R.min(), pred_score - target[i])
            pred_loss += torch.abs(pred_score - target[i])#F.mse_loss(pred_score, target[i].int())
            R_loss -= abs(R).mean()
#         print(R_loss,pred_loss)
        total_loss = pred_loss + R_loss
#         print()
        clip_value = 0.1
#         for p in sigma_model.parameters():
#             p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        total_loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))
#         break


# In[49]:


sigma_model = FCN().to(device)
optimizer = optim.SGD(sigma_model.parameters(), lr=0.0001, momentum=momentum)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
lr = .01
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
checkpoint = torch.load("trained_networks/trained_starter")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
model.eval()

sigma_model = FCN().to(device)
optimizer = optim.SGD(sigma_model.parameters(), lr=0.0000001, momentum=momentum)
# print(torch.abs(sigma_model(inputs)).view(inputs.shape)[0].shape)

l1loss = nn.L1Loss()
certified_model = Smooth(model) #,torch.abs(sigma_model(inputs)).view(inputs.shape)[0])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True, **kwargs)

train_sigma_network(sigma_model,certified_model,device,train_loader,optimizer,epoch,log_interval)


# In[ ]:


import torchvision
import matplotlib.pyplot as plt
sigma = sigma_model(inputs).view(inputs.shape)
# print(sigma)
torchvision.utils.save_image(sigma,'visual_sigma.png')


# In[ ]:




