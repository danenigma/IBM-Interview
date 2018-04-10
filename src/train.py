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
    
def validate(cnn, fc, data_loader, criterion):
	val_size = len(data_loader)
	val_loss = 0
	cnn.eval()
	fc.eval()
	for i, (images, labels) in enumerate(data_loader):
		# Set mini-batch dataset
		images    = to_var(images, volatile=True)
		labels    = to_var(labels)
		outputs   = fc(cnn(images))
		loss = criterion(outputs, labels)
		val_loss += loss.data.sum()
		
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
	cnn = ResNet()
	fc  = FC()
	 
	try:
		fc.load_state_dict(torch.load(args.model_path))
		print("using pre-trained model")
	except:
		print("using new model")
	criterion = nn.CrossEntropyLoss()
	if torch.cuda.is_available():
		fc.cuda()
		cnn.cuda()
		criterion.cuda()
	

	optimizer  = torch.optim.Adam(fc.parameters(), lr=args.learning_rate)
	total_step = len(train_loader)
	print('validating.....')
	best_val = validate(cnn, fc, val_loader, criterion)
	print("starting val loss {:f}".format(best_val))
	
	for epoch in range(args.num_epochs):
		fc.train()
		cnn.train()
		for i, (images, labels) in enumerate(train_loader):

			# Set mini-batch dataset
			images    = to_var(images, volatile=True)
			labels    = to_var(labels)
			# Forward, Backward and Optimize
			fc.zero_grad()
			cnn.zero_grad()
			
			outputs = fc(cnn(images))

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Print log info
			if i % args.log_step == 0:
				print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
					  %(epoch, args.num_epochs, i, total_step, 
						loss.data[0])) 

			# Save the models
		if (epoch+1) % args.save_step == 0:
			val_loss = validate(cnn, fc, val_loader, criterion)
			print('val loss: ', val_loss)
			
			if val_loss < best_val:
				best_val = val_loss
				fc_cpu   = fc.cpu()
				print("Found new best val")
				torch.save(fc_cpu.state_dict(), 
					   args.model_path)
				if torch.cuda.is_available():
					fc.cuda()	   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/train/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=2,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--model_path', type=str, default='../models/fc_best.pt',
                        help='path for trained encoder')

    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)
