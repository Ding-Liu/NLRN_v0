# Non-Local Recurrent Network for Image Restoration (NeurIPS 2018)

[Paper](http://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf) | [Bibtex](#Bibtex)

This is the original version of the NLRN code. A newer version can be found [here](https://github.com/Ding-Liu/NLRN).

## Usage
### Denoising
#### Preparing 400 images from BSDS500 for training
Under the root directory of this repository
```
mkdir -p data
```
Download the compressed 400 training images in grayscale [here](https://drive.google.com/file/d/1vO65GolyMqUNc3_K9h-dHLiWZcZP2g9E/view?usp=sharing). They are converted from the color images in [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html). Move the compressed file `BSDS500.tar.gz` to `./data`
Uncompress them and generate the training file name list:
```
cd data
tar -zxf BSDS500.tar.gz
cd BSDS500
find train_gray_rgb2gray/*.png test_gray_rgb2gray/*.png > ../train.list
```
#### Preparing Set12 and BSD68 for evaluation
These two datasets can be downloaded from [here](https://github.com/cszn/DnCNN.git). Move them to `./data` 
#### Training on 400 images (train and test) of BSD500
Under the root directory of this repository
```
bash train.sh
```
#### Released models
Noise level (sigma): [15](https://drive.google.com/file/d/19SoZaee_7kFzwnLQvY0uR9NbQl0ddSPG/view?usp=sharing) [25](https://drive.google.com/file/d/1uvD4MuaK1DpZugnlybAccjxvVe1O0jFO/view?usp=sharing) [50](https://drive.google.com/file/d/1Q6nLh4dAvDDEfCzWu6C_GUIrn4paKw9W/view?usp=sharing)
#### Prediction on Set12 and BSD68
Unzip the downloaded files and move them under `./checkpoints`
```
bash test.sh
```

### Image Super-Resolution

#### Released models
The model can be downloaded [here](https://drive.google.com/file/d/1VWa_IcYClCMaaPqiWNzgPRJfaTB-zSni/view?usp=sharing).
#### Preparing Set5 and Set14 for evaluation
These two datasets can be downloaded from [here](https://github.com/cszn/DnCNN.git). Move them to `./data`
#### Prediction on Set5 and Set14
Unzip the downloaded files and move them under `./checkpoints`
```
bash test_sr.sh
```

## Dependencies
- Python 2.7
- TensorFlow 1.10
## Bibtex
```
@inproceedings{liu2018non,
  title={Non-Local Recurrent Network for Image Restoration},
  author={Liu, Ding and Wen, Bihan and Fan, Yuchen and Loy, Chen Change and Huang, Thomas S},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1680--1689},
  year={2018}
}
```
