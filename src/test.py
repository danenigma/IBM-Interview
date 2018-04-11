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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

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