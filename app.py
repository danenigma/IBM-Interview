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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
model_path = 'models/resnet50_best.pt'
#cnn   = ResNet()
#fc    = FC()
model = ResNet50()
#print(fc)
#model = ResNetCNN()
try:
	model.load_state_dict(torch.load(model_path))
	model.eval()
	print('Model Loading Done!!')
except:
	print("Model Loading Failed!!")
	
classes = {
		 0: 'Black-grass', 
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
         
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

@app.route('/')
def hello_world():
	print("got request")
	return render_template('index_new.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
	print('android: ', request)
	return render_template('index_new.html')
	
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['image']
		f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(f)
		print(f)
		image  = Image.open(f).convert('RGB')
		image  = transform(image).unsqueeze(0)
		print(image.shape)
		
		#out    = fc(cnn(Variable(image)))
		out    = model(to_var(image, volatile=True))
		print(out.data)
		pred   = out.data.max(1, keepdim=True)[1].int()
		print("out: ", classes[int(pred[0])])	
		return classes[int(pred[0])]
	return None
if __name__ == '__main__':
    app.run(debug=True)
