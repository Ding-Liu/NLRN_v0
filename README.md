# Non-Local Recurrent Network for Image Restoration

[Paper](http://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf) | [Bibtex](#Bibtex)

This is the original version of the NLRN code.

## Usage
### Denoising
#### Preparing 400 training images from BSDS500
Under the directory of this repository
```
mkdir -p training_data
```
Download the compressed 400 training images from BSDS500 [here](https://drive.google.com/file/d/1vO65GolyMqUNc3_K9h-dHLiWZcZP2g9E/view?usp=sharing). Move the compressed file to `./training_data`
```
cd training_data
tar -zxf BSDS500.tar.gz
cd BSDS500
find train_gray_rgb2gray/*.png test_gray_rgb2gray/*.png > train.list
```
#### Released models
Noise level (sigma): [15](https://drive.google.com/file/d/19SoZaee_7kFzwnLQvY0uR9NbQl0ddSPG/view?usp=sharing) [25](https://drive.google.com/file/d/1uvD4MuaK1DpZugnlybAccjxvVe1O0jFO/view?usp=sharing) [50](https://drive.google.com/file/d/1Q6nLh4dAvDDEfCzWu6C_GUIrn4paKw9W/view?usp=sharing)

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
