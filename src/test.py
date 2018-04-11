import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from models import *
from torch.autograd import Variable 
from torchvision import transforms
from data_loader import *

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def test(model, test_loader, test_table):
    model.eval()
    names = {0: 'Black-grass', 
             1: 'Charlock', 
             2: 'Cleavers', 
             3: 'Common Chickweed', 
             4: 'Common wheat', 
             5: 'Fat Hen', 
             6: 'Loose Silky-bent', 
             7: 'Maize', 
             8: 'Scentless Mayweed', 
             9: 'Shepherds Purse', 
             10: 'Small-flowered Cranesbill', 
             11: 'Sugar beet'}
    
    for i, (images, labels) in enumerate(test_loader):
        images    = to_var(images, volatile=True)
        outputs   = model(images)
        pred = outputs.data.max(1, keepdim=True)[1].int()
        test_table.iloc[i, 2] = int(pred[0])
        test_table.iloc[i, 1] = names[int(pred[0])]
    test_table.to_csv('sub.csv')
    print('Done!!')
def validate(model, test_loader, test_table):
    model.eval()
    names = {0: 'Black-grass', 
             1: 'Charlock', 
             2: 'Cleavers', 
             3: 'Common Chickweed', 
             4: 'Common wheat', 
             5: 'Fat Hen', 
             6: 'Loose Silky-bent', 
             7: 'Maize', 
             8: 'Scentless Mayweed', 
             9: 'Shepherds Purse', 
             10: 'Small-flowered Cranesbill', 
             11: 'Sugar beet'}
    out = []
    for i, (images, labels) in enumerate(test_loader):
        images    = to_var(images, volatile=True)
        outputs   = model(images)
        pred = outputs.data.max(1, keepdim=True)[1].int()
        out.append([int(pred[0]), int(labels[0])])
        #print(int(pred[0]), int(labels[0]))
        #test_table.iloc[i, 2] = int(pred[0])
        #test_table.iloc[i, 1] = names[int(pred[0])]
    #test_table.to_csv('sub.csv')
    np.save('pred.npy', np.array(out))
    print('Done!!')
    
    
if __name__=='__main__':
    img_size = 224

    test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_ds     = IBMDataset('../data/test/', transform=test_trans, name='test')
    test_table  = test_ds.table
    test_loader = data.DataLoader(test_ds, 
                             batch_size = 1,
                             shuffle = False)
    val_ds     = IBMDataset('../data/train/', transform=test_trans, name='val')
    val_table  = val_ds.table
    val_loader = data.DataLoader(val_ds, 
                             batch_size = 1,
                             shuffle = False)

    model =  ResNet50()
    model.load_state_dict(torch.load('../models/resnet50_best.pt'))    
    if torch.cuda.is_available():
        model.cuda()
    #test(model, test_loader, test_table)
    validate(model, val_loader, test_table)
