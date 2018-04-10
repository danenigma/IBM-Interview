import os
import numpy as np # linear algebra
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision
from torchvision import transforms
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from models import *
from PIL import Image
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

class IBMDataset(data.Dataset):

	def __init__(self, data_dir='../data', name = 'train', transform=None):
		self.table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pickle'.format(name)))
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):
		img_name = self.table.iloc[idx, 0]
		fullname = os.path.join(self.data_dir, img_name)
		image  = Image.open(fullname).convert('RGB')
		if self.transform:
			image = self.transform(image)
			
		label  = int(self.table.iloc[idx, 2])

		return image, label
		
def my_collate(batch):
	images, labels = zip(*batch)
	print(labels)
	
	dict_ = {
		"images":torch.cat(images, dim=0),
		"labels":torch.LongTensor(labels)
		}
	return dict_

def save_model(model, model_path):
	torch.save(model.state_dict(),model_path)


if __name__ == '__main__':
	data_dir    = '../data/train'

	img_size    = 224 

	train_trans = transforms.Compose([
		transforms.RandomResizedCrop(img_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_trans = transforms.Compose([
    	transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_ds     = IBMDataset(data_dir, transform=train_trans, name='train')
	val_loader = data.DataLoader(val_ds, 
								 batch_size = 8,
								 shuffle = True)

	print(len(val_ds))
	model = ResNetCNN()
	for i, (img, labels) in enumerate(val_loader):
		print(img.shape, labels)
		print(model(Variable(img)).shape)
		break
	
	save_model(model, '../models/resnet_test.pt')
