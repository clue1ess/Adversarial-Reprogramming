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


'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


class CIFAR10(): 
    def __init__(self, batch_size = 100, num_epochs = 50, learning_rate = 0.01, momentum = 0.9, step_size = 12, gamma = 0.2):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.step_size = 12
        self.gamma = 0.2
        
    def CrossEntropyLoss(self) :
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def train_dataloader(self):
        '''Loading training dataset'''
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.),(1.,1.,1.))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                           train=True, 
                                           transform=train_transform,
                                           download=True)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = self.batch_size, 
                                           shuffle = True)
        return train_loader

    def val_dataloader(self):
        '''Loading validation dataset'''
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.),(1.,1.,1.))
        ])
        val_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                           train=True, 
                                           transform=val_transform,
                                           download=True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = self.batch_size, 
                                           shuffle = True)
        return val_loader
    
    def test_dataloader(self):
        '''Loading testing dataset'''
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.),(1.,1.,1.))
        ])
        test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                          train=False, 
                                          transform=test_transform)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = self.batch_size, 
                                          shuffle = False)
        return test_loader
    
    def training_step(self) :
        '''Training CIFAR10 dataset on Resnet18 model'''
        train_dataloader = self.train_dataloader()  #loading training data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #loading resnet18 model
        model = ResNet18()
        model = model.to(device)               
        
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, momentum = self.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        criterion = self.CrossEntropyLoss()
        
        with torch.set_grad_enabled(True):
            model.train()
            list_accuracy = []
            list_loss = []
            for epoch in range(self.num_epochs) :
                total_accuracy = 0
                i = 0
                for _,data in enumerate(train_dataloader) :
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_accuracy += (torch.max(outputs,1)[-1] == labels).sum().item()
                    i = i + self.batch_size
                accuracy = (total_accuracy / i) * 100
                loss = 100 - accuracy
                list_accuracy.append(accuracy)
                list_loss.append(loss)
                if epoch > 40: 
                    filename = "resnet_" + str(epoch) +".pt"
                    torch.save(model, filename)
                print("epoch " + str(epoch) + " -> " + "accuracy : " + "{:.2f}".format(accuracy) + " loss : " + "{:.2f}".format(loss))
                scheduler.step()
            print("Finished!")
        #torch.save(model.state_dict(), '../save_model')
        

        #plotting graphs 
        #accuracy vs epoch
        x = np.arange(0, self.num_epochs)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy and loss")
        plt.plot(x, list_accuracy)
        plt.plot(x, list_loss)
    
    def validate(self) :
        '''validation Phase'''
        #loading the model with trained parameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_dataloader = self.val_dataloader()
        criterion = self.CrossEntropyLoss()
        start_ = self.num_epochs - 9
        end_ = self.num_epochs
        accuracy_val = []
        
        for epoch in range(start_, end_) :
            filename = "resnet_" + str(epoch) + ".pt"
            model = torch.load(filename)
            model.eval()
            total_accuracy = 0
            i = 0
            with torch.set_grad_enabled(False):
                for _,data in enumerate(val_dataloader) :
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    loss = criterion(outputs, labels)
                    total_accuracy += (torch.max(outputs,1)[-1] == labels).sum().item()
                    i = i + self.batch_size
                    accuracy = (total_accuracy / i)*100
                    print("epoch : " + str(epoch) + " accuracy on val set : " + str(accuracy))
                    accuracy_val.append(accuracy)
        print(accuracy_val)
       
        
    def testing_step(self) :
        '''Testing Phase'''
        #loading the model with trained parameters
        print(accuracy_val)
        index = accuracy_val.index(max(accuracy_val))
        index  = index + self.num_epochs - 9
        filename = "resnet_" + str(index) + ".pt"
        print("filename : " + filename)
        model = torch.load(filename)
        model.eval()

        #loading testing dataset
        test_dataloader = self.test_dataloader()
        criterion = self.CrossEntropyLoss()
        
        total_accuracy = 0
        i = 0
        with torch.set_grad_enabled(False):
            for _,data in enumerate(test_dataloader) :
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                outputs = outputs.to(device)
                loss = criterion(outputs, labels)
                total_accuracy += (torch.max(outputs,1)[-1] == labels).sum().item()
                i = i + self.batch_size
            print("accuracy on test set : " + str((total_accuracy / i)*100))


def main(): 
    if(len(sys.argv) == 7):
        model = CIFAR10(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
        model.training_step()
        model.validate()
        model.testing_step()

    elif(len(sys.argv) == 1):
        model = CIFAR10()
        model.training_step()
        model.validate()
        model.testing_step()
        
    else:
        print("Incomplete arguments")

    
if __name__ == '__main__':
    main()
    