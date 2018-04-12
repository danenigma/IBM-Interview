# Seedling Species Identification Web App
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/web.png)

### Description
Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.
![alt tag](https://github.com/danenigma/IBM-Interview/blob/master/seedlings.png)
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

