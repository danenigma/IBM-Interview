import torch
import torchvision
from torchvision import transforms
import numpy as np
import os
from models import *
from torch.autograd import Variable
from PIL import Image
classes= {
		  0: 'Common wheat',
		  1: 'Sugar beet',
		  2: 'Scentless Mayweed',
		  3: 'Black-grass', 
		  4: 'Small-flowered Cranesbill',
		  5: 'Maize',
		  6: 'Charlock', 
		  7: 'Common Chickweed', 
		  8: 'Loose Silky-bent', 
		  9: 'Fat Hen', 
		  10: 'Cleavers', 
		  11: 'Shepherds Purse'
		  }
		  
    
model_path = '../models/resnet_best.pt'
model = ResNetCNN()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model.load_state_dict(torch.load(model_path))
#fc  = model.fc
#cnn = ResNet()
#print(fc)
f = "../data/train/Sugar beet/fc293eacb.png"#
image  = Image.open(f).convert('RGB')
image  = transform(image).unsqueeze(0)
out    = model(Variable(image))
pred   = int(out.data.max(1, keepdim=True)[1].numpy()[0])
print("out: ", classes[pred])	
#for layer in fc:
#	out = layer(out)	
#print(out.shape)
#fc_path = '../models/fc_best.pt'
#fc = fc.cpu()
#torch.save(fc.state_dict(), fc_path)
#print("saving done!!")


