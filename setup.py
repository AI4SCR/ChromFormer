# configuration approach followed:
# - whenever possible, prefer pyproject.toml
# - for configurations insufficiently supported by pyproject.toml, use setup.cfg instead
# - setup.py discouraged; minimal stub included only for compatibility with legacy tools

# Minimal stub for backwards compatibility, e.g. for legacy tools without PEP 660 support.
# See https://setuptools.pypa.io/en/latest/userguide/quickstart.html
from setuptools import setup
from setuptools import find_packages

#setup(packages = ['Chromatin3D', 'Chromatin3D.data_generation', 'Chromatin3D.Data_Tools'])
setup()