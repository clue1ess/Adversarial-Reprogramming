#!/usr/bin/python
#Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

class Reprogramming(torch.nn.Module) :
    def __init__(self, model, orginal_size = 32, adversarial_size = 14, batch_size = 100, lr=0.1, epoch=49) :
        super(Reprogramming, self).__init__()
        self.model = model
        self.orginal_size = orginal_size
        self.adversarial_size = adversarial_size
        self.batch_size = batch_size
        self.weights = torch.nn.Parameter(torch.randn(3, self.orginal_size, self.orginal_size), requires_grad=True)
        org_center = self.orginal_size // 2
        adv_center = self.adversarial_size // 2
        if(adversarial_size % 2) :
            self.lower = org_center - adv_center - 1
        else :
            self.lower = org_center - adv_center
        self.upper = org_center + adv_center
        self.mask = torch.randn(3, self.orginal_size, self.orginal_size)
        self.mask[:,self.lower:self.upper, self.lower:self.upper] = 0
        mean = np.array([0, 0, 0]).reshape(1, 3, 1, 1)
        std = np.array([1, 1, 1]).reshape(1, 3, 1, 1)
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.mean, self.std = self.mean.type(torch.FloatTensor), self.std.type(torch.FloatTensor)
  
    def forward(self, x) :
        temp = self.weights * self.mask
        #parameters = torch.tanh(temp)
        parameters = torch.sigmoid(temp)
        parameters = parameters.to(device)
        image = torch.ones(self.batch_size, 3, self.orginal_size, self.orginal_size)
        for index in range(self.batch_size) :
            image[index] = parameters
        for i in range(3) :
            image[:, i, self.lower:self.upper, self.lower:self.upper] = x[:, 0, :, :]
        x = (image - self.mean) / self.std
        x = x.to(device)
        y = self.model(x)
        return y
    
    class MNIST(): 
    def __init__(self, batch_size = 100, num_epochs = 50, learning_rate = 0.1, momentum = 0.9, img_size = 14, filename):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.filename = filename
        
    def CrossEntropyLoss(self) :
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def train_dataloader(self):
        mnist_transform = torchvision.transforms.Compose([transforms.Resize((self.img_size,self.img_size)), transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True, 
                                           transform=mnist_transform,
                                           download=True)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = self.batch_size, 
                                           shuffle = True)
        return train_loader
    
    def test_dataloader(self):
        mnist_transform = torchvision.transforms.Compose([transforms.Resize((self.img_size,self.img_size)), transforms.ToTensor()])
        test_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=False, 
                                           transform=mnist_transform,
                                           download=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = self.batch_size, 
                                           shuffle = True)
        return test_loader
    
    def training_step(self) :

        #loading model
        train_dataloader = self.train_dataloader()
        model = torch.load(self.filename)
        model.eval()
        reprogrammed = Reprogramming(model, adversarial_size = self.img_size) 
        #parameters = reprogrammed.weights
        #parameters = parameters.to(device)
        optimizer = torch.optim.SGD([reprogrammed.weights], lr = self.learning_rate, momentum = self.momentum)
        criterion = self.CrossEntropyLoss()

        #training phase
        with torch.set_grad_enabled(True):
            list_accuracy = []
            list_loss = []
            for epoch in range(self.num_epochs) :
                total_accuracy = 0.0
                i = 0
                for _, data in enumerate(train_dataloader) :
                    (inputs, labels) = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = reprogrammed.forward(inputs)
                    outputs = outputs.to(device)
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_accuracy += (torch.max(outputs,1)[-1] == labels).sum().item()
                    i = i + self.batch_size
                accuracy = (total_accuracy / i) * 100
                loss = 100 - accuracy
                list_accuracy.append(accuracy)
                list_loss.append(loss)
                print("epoch " + str(epoch) + " -> " + "accuracy : " + "{:.2f}".format(accuracy) + " loss : " + "{:.2f}".format(loss))
        print("Finished!")

          #plotting graphs 
          #accuracy vs epoch
          x = np.arange(0, self.num_epochs)
          plt.xlabel("Number of epochs")
          plt.ylabel("Accuracy and loss")
          plt.plot(x, list_accuracy)
          plt.plot(x, list_loss)

        #testing phase
        test_dataloader = self.test_dataloader()
        total_accuracy = 0.0
        i = 0
        with torch.set_grad_enabled(False):
            for _,data in enumerate(test_dataloader) :
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = reprogrammed(inputs)
                outputs = outputs.to(device)
                loss = criterion(outputs, labels)
                total_accuracy += (torch.max(outputs,1)[-1] == labels).sum().item()
                i = i + self.batch_size
            print("accuracy on test set : " + str((total_accuracy / i)*100))
            
            
def main(): 
    if(len(sys.argv) == 7):
        model = MNIST(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
        model.training_step()

    elif(len(sys.argv) == 1):
        model = CIFAR10()
        model.training_step()

    else:
        print("Incomplete arguments")

    
if __name__ == '__main__':
    main()
    