# Seedling Species Identification Web App
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/web.png)

## Description
Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/seedlings.png)

## Results


#### Dataset

The dataset (~2Gb) comprises 12 images of plant species at several growth stages.

| Species Names |
| ------------- |
| Black-grass |
| Charlock |
| Cleavers |
| Common Chickweed |
| Common wheat |
| Fat Hen | 
| Loose Silky-bent |
| Maize |
| Scentless Mayweed |
| Shepherds Purse |
| Small-flowered Cranesbill |
| Sugar beet |

You can obtain the original dataset [here](https://vision.eng.au.dk/plant-seedlings-dataset/)
#### Models 

I experimented with pre-trained CNN models with their final FC layers removed, Resnet50 and Resnet152, to extract useful features for classification.

After 20 epochs of training on a 80-20 spit of the dataset, 80% training 20% validation, I got the following result.


| Model    | Accuracy (%) | Size (Mb) |
| -------- | -------- | ---------- |
| Resnet50  single layer FC| 80.1 | 102 |
| Resnet153 three layer FC| 86.8 | 228 |
 
##### Confusion Matrix for Resnet50 with a single FC layer 
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/Figure_1.png)
##### Confusion Matrix for Resnet152 with three FC layer 
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/Figure_2.png)

### Installation

Install [conda](https://conda.io/docs/user-guide/install/index.html) and install dependencies

```sh
$ cd ~
$ git clone https://github.com/danenigma/IBM-Interview.git
$ cd ~/IBM-Interview/
$ conda install --yes --file requirements.txt
```
## Usage
#### Training Model
```sh
$ cd ~/IBM-Interview/
$ python train.py --data_dir data --num_epochs 10 --batch_size 32 --learning_rate 0.001 
```
#### Running Locally
```sh
$ cd ~/IBM-Interview/
$ python app.py
To view the result go to 127.0.0.1:5000/
```
#### View Demo
visit this [link](http://ec2-35-155-215-84.us-west-2.compute.amazonaws.com:5000/)

you can use these [test images](https://github.com/danenigma/IBM-Interview/blob/master/test_images/)
to test the system 
## Use Case Diagram
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/usecase.png)


### References

	@article{DBLP:journals/corr/abs-1711-05458,
	  author    = {Thomas Mosgaard Giselsson and
		           Rasmus Nyholm J{\o}rgensen and
		           Peter Kryger Jensen and
		           Mads Dyrmann and
		           Henrik Skov Midtiby},
	  title     = {A Public Image Database for Benchmark of Plant Seedling Classification
		           Algorithms},
	  journal   = {CoRR},
	  volume    = {abs/1711.05458},
	  year      = {2017},
	  url       = {http://arxiv.org/abs/1711.05458},
	  archivePrefix = {arXiv},
	  eprint    = {1711.05458},
	  timestamp = {Fri, 01 Dec 2017 14:22:24 +0100},
	  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1711-05458},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	}


