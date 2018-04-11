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
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def validate(model, data_loader, criterion):
	val_size = len(data_loader)
	val_loss = 0
	#cnn.eval()
	#fc.eval()
	model.eval()
	correct = 0
	for i, (images, labels) in enumerate(data_loader):
		# Set mini-batch dataset
		images    = to_var(images, volatile=True)
		labels    = to_var(labels)
		#outputs  = fc(cnn(images))
		outputs   = model(images)
		pred = outputs.data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(labels.data.view_as(pred).int())
		correct += predicted.sum()
		loss = criterion(outputs, labels)
		val_loss += loss.data.sum()
	print('val acc: ', correct/950)
	return val_loss/val_size

def main(args):

    img_size = 224
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds     = IBMDataset(args.data_dir, transform=train_trans, name='train')
    train_loader = data.DataLoader(train_ds, 
                                 batch_size = args.batch_size,
                                 shuffle = True)

    val_ds     = IBMDataset(args.data_dir, transform=test_trans, name='val')
    print(len(val_ds), len(train_ds))
    val_loader = data.DataLoader(val_ds, 
                                 batch_size = args.batch_size,
                                 shuffle = True)
    LOG_DIRECTORY = '../logs/'
    SAVE_DIRECTORY = '../models/resnet50/'
    
    model =  ResNet50()
    trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
    .validate_every((1, 'epochs')) \
    .save_every((1, 'epochs')) \
    .save_to_directory(SAVE_DIRECTORY) \
    .set_max_num_epochs(args.num_epochs) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                  log_images_every='never'),
                log_directory=LOG_DIRECTORY)

    # Bind loaders
    trainer \
    .bind_loader('train', train_loader) \
    .bind_loader('validate', val_loader)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/train/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=2,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--model_path', type=str, default='../models/resnet_best.pt',
                        help='path for trained encoder')

    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)
