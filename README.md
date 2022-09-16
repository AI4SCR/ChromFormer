# 3D Chromatin Prediction

This project aims to predict the shape of the Chromatin using Hi-C interaction matrices as input

## Environment
```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## Installation

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
## Usage
...

## Contributing

Check [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Getting support

Check [SUPPORT.md](.github/SUPPORT.md).

## Credits
This project was created using https://github.ibm.com/HCLS-innersource/python-blueprint.
