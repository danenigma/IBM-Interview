import torch
from models import *
from torch.autograd import Variable

model_path = '../models/fc_best.pt'
#model = ResNetCNN()
fc = FC()
fc.load_state_dict(torch.load(model_path))
#fc  = model.fc
cnn = ResNet()
print(fc)
x_test = torch.randn(1,3,224,224)
out = fc(cnn(Variable(x_test)))

#for layer in fc:
#	out = layer(out)	
#print(out.shape)
#fc_path = '../models/fc_best.pt'
#fc = fc.cpu()
#torch.save(fc.state_dict(), fc_path)
#print("saving done!!")


