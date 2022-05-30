#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_packages, find_namespace_packages

import sys

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements
requirements = []
with open(os.path.join(HERE, 'requirements.txt'), 'r') as fh:
    for line in fh:
        if not line.startswith('#'):
            requirements.append(line.strip())

# version
version_file = open(os.path.join(HERE, 'VERSION.txt'))
version = version_file.read().strip()

setup(name='fuse-med-ml',
      version=version,
      description='Open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/IBM/fuse-med-ml/',
      author='IBM Research - Machine Learning for Healthcare and Life Sciences',
      author_email='moshiko.raboh@ibm.com',
      packages=find_namespace_packages(),
      license='Apache License 2.0',
      install_requires=requirements
      )
