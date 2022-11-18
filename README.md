# 3D Chromatin Prediction

This project aims to predict the shape of the Chromatin using Hi-C interaction matrices as input

## Environment
```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## Installation
brew install pyenv
pyenv install 3.8 
pyenv local 3.8.15
brew install pyenv-virtualenv 
eval "$(pyenv init -)"         
eval "$(pyenv virtualenv-init -)"
pyenv virtualenv 3.8.15 chromenv
pyenv local chromenv 

## Package installation
```sh
# assuming you have an SSH key set up on GitHub
pip install "git+ssh://git@github.ibm.com/AI4SCR-DEV/3D-Chromatin.git@main"
```

### Suggested setup for development
```sh
pip install -r requirements.txt
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pre-commit install
```
## Package Documentation
```sh
```
Find Documentation for Code [here](https://pages.github.ibm.com/AI4SCR-DEV/3D-Chromatin/)


