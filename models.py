__author__ = 'Daniel Marew'

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class ResNetCNN(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNetCNN, self).__init__()
        resnet  = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet  = nn.Sequential(*modules)
        self.fc      = nn.ModuleList([
        			   nn.Linear(resnet.fc.in_features, 1024),
        			   nn.ReLU(),
        			   nn.Linear(1024, 1024),
        			   nn.ReLU(),
        			   nn.Linear(1024, 12)])
        
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        for layer in self.fc:
        	features = layer(features)
        return features
class ResNet50(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResNet50, self).__init__()
        resnet  = models.resnet50(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet  = nn.Sequential(*modules)
        self.fc      = nn.ModuleList([
        			   nn.Linear(resnet.fc.in_features, 1024),
        			   nn.Linear(1024, 12)])
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        for layer in self.fc:
        	features = layer(features)
        return features

    
class ResNet(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        resnet  = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet  = nn.Sequential(*modules)
        
        print('cnn feat: ', resnet.fc.in_features)
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        return features
        
class FC(nn.Module):

    def __init__(self):
        super(FC, self).__init__()
        self.fc      = nn.ModuleList([
        			   nn.Linear(2048, 1024),
        			   nn.ReLU(),
        			   nn.Linear(1024, 1024),
        			   nn.ReLU(),
        			   nn.Linear(1024, 12)])
    
        
    def forward(self, features):
        """FC layers"""
        for layer in self.fc:
        	features = layer(features)
        return features

if __name__=='__main__':
	resnet = ResNetCNN(1)
	print(resnet)
	x_test = torch.randn(8,3,224,224)
	print(resnet(Variable(x_test)))
	
