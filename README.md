# plancklens

[![alt text](https://readthedocs.org/projects/plancklens/badge/?version=latest)](https://plancklens.readthedocs.io/en/latest)[![Build Status](https://travis-ci.com/carronj/plancklens.svg?branch=master)](https://travis-ci.com/carronj/plancklens)

plancklens is is a python code for cosmology containing most of Planck 2018 CMB lensing pipeline, by Julien Carron on behalf of the *Planck* collaboration ([publication here.](https://arxiv.org/abs/1807.06210))
Some numerical parts are written in Fortran. Portions of it (structure and code) have been directly adapted from pre-existing work by Duncan Hanson.

### Installation

After cloning the repository, build an editable installation with
    
    pip install -e . [--user]

The –-user is required only if you don’t have write permission to your main python installation. A fortran compiler is required for a successful installation.

### Contents

This code contains most of the Planck 2018 lensing pipeline. In particular it possible to reproduce the published map and band-powers basically exactly. 

Some parts of the pipeline have been left out or are not yet translated to python 3. This is the case notably of the band-powers likelihood code, or the code used to produce lensed CMB skies (the latter code is the stand-alone package [lenspyx](https://github.com/carronj/lenspyx))

### Example parameter files

To use the examples lensing reconstruction parameter files, you will need further to define the environment variable $PLENS to some place safe to write.
    
Details on the structure of a parameter file are given in this one: [idealized_example.py](params/idealized_example.py)

In order to reproduce the 2018 lensing maps and spectrum band-powers, one may use the provided [smicadx12_planck2018.py](params/smicadx12_planck2018.py) parameter file.


The basics on how to use parameter files can be found in [this jupyter notebook](examples/lensingrec_quickstart.ipynb).
Some details on the numerical computations are collected [in this document](https://arxiv.org/abs/1908.02016).
You might also need to check out the [plancklens documentation](https://plancklens.readthedocs.io/en/latest) (with some bits in progress).

* Jan 14 2021: important bug fix in fortran N1 file affecting EB TB and TE. The n1 module must be rebuilt.

![ERC logo](https://erc.europa.eu/sites/default/files/content/erc_banner-vertical.jpg)
