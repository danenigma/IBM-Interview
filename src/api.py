import os
import torchvision
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request
from models import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable 
app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_path = '../models/resnet_best.pt'
#cnn   = ResNet()
#fc    = FC()
model = ResNetCNN()
#print(fc)
#model = ResNetCNN()
try:
	#l = torch.load(model_path)
	#print("l: ", l)
	#fc.load_state_dict(l)
	model.load_state_dict(torch.load(model_path))
	print('Model Loading Done!!')
except:
	print("Model Loading Failed!!")
	
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
		  


@app.route('/')
def hello_world():
    return render_template('index_new.html')
    
@app.route('/home')
def home():
	return render_template('index_new.html')
	
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['image']
		f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(f)
		image  = Image.open(f).convert('RGB')
		image  = transform(image).unsqueeze(0)
		#out    = fc(cnn(Variable(image)))
		out    = model(Variable(image))
		print(out.data)
		pred   = int(out.data.max(1, keepdim=True)[1].numpy()[0])
		print("out: ", classes[pred])	
		return classes[pred]
	return None
if __name__ == '__main__':
    app.run(debug=True)
