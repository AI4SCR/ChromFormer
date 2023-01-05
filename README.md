# ChromFormer

This project aims to predict the shape of the Chromatin using Hi-C interaction matrices as input. It implements the following [paper](https://www.biorxiv.org/content/10.1101/2022.11.15.516571v1).

## Environment
First, install python 3.8. If python 3.8 is already installed, you can use 
```sh
python3 -m venv chromenv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```
To create your virtual environment.
If python 3.8 is not installed the following can be done:
### Mac
```
brew install pyenv
pyenv install 3.8 
pyenv local ["full version of python that got install"]
brew install pyenv-virtualenv 
eval "$(pyenv init -)"         
eval "$(pyenv virtualenv-init -)"
pyenv virtualenv ["full version of python that got install"] chromenv
pyenv local chromenv 
```
### Windows
```
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME\.pyenv"

[System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

[System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

[System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

[System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")

pyenv install ["compatible 3.8 windows version"]

pyenv local ["full version of python that got install"]

python -m pip install -user virtualenv

python -m venv chromenv

pyenv local chromenv

.\chromenv\Scripts\activate
```
## Package installation
Then you will need to install the ChromFormer package
```sh
pip install "git+ssh://git@github.com:AI4SCR/ChromFormer.git"
```

### Suggested setup for development
```sh
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html
pre-commit install
```
## Package Documentation

Find Documentation for Code [here](https://pages.github.ibm.com/AI4SCR-DEV/3D-Chromatin/)

## Code organisation

The Following is the organisation of the repository:

```ChromFormer``` is the package that contains the ```Uniform_Cluster_Walk.py``` data_generation file, ```Data_Tools``` that consist of optimal transport, plots and calculations needed for the Model and Data generation process.

```bench``` consists of all experiments attempted such as data generation, model training and comparision with previous works. To use jupyter notebooks in the bench you will need to create a .env file at the root of the repository with variable DATA_DIR="your path to the data file". Directions on further organisation of the bench and which files to run are given in the README of the bench folder. A demo of an end to end example is given in the demo folder inside the bench.  

```data``` folder contains the generated synthetic and imported data 


## Installation
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.1+cpu.html
pip install -e .
```