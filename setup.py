# -*- coding: utf-8 -*-

from setuptools import setup
import re

## from https://gehrcke.de/2014/02/distributing-a-python-command-line-application/
## note that it requires double quotes around the version number... should fix!
version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('realization/__init__.py').read(),
    re.M
    ).group(1)

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(name='realization',
      version=version,
      description='Realizations and spectra of n-dim Gaussian random fields.',
      url='http://github.com/defjaf/aniso',
      author='Andrew H. Jaffe',
      author_email='a.h.jaffe@gmail.com',
      install_requires=['numpy', 'matplotlib', 'scipy'],
      packages=['realization']
)

####  testing reminder 
# virtualenv --python=python2 venvpy27
# source venvpy27/bin/activate
# python setup.py install

# virtualenv --python=python3 venvpy3
# source venvpy3/bin/activate
# python setup.py install
