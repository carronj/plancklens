# plancklens

[![alt text](https://readthedocs.org/projects/plancklens/badge/?version=latest)](https://plancklens.readthedocs.io/en/latest)[![Build Status](https://travis-ci.com/carronj/plancklens.svg?branch=master)](https://travis-ci.com/carronj/plancklens)

plancklens is is a python code for cosmology containing most of Planck 2018 CMB lensing pipeline, by Julien Carron on behalf of the *Planck* collaboration ([publication here.](https://arxiv.org/abs/1807.06210))
Some numerical parts are written in Fortran. Portions of it (structure and code) have been directly adapted from pre-existing work by Duncan Hanson.

This package may be used for:
* lensing (as well as other types of anisotropies) reconstruction on actual *Planck* data or other data/sims 
* or simply for calculation of responses, reconstruction noise levels and biases for forecasts or other analytical work

### Installation

After cloning the repository, build an editable installation with
    
    pip install -e . [--user]

The –-user is required only if you don’t have write permission to your main python installation. A fortran compiler is required for a successful installation.

### Contents

This code contains most of the Planck 2018 lensing pipeline. In particular it possible to reproduce the published map and band-powers basically exactly. Some more detailed parts of the pipeline have been left out or are not yet translated to python 3. This is the case notably of the band-powers likelihood code.

The code used to produce lensed CMB skies is the stand-alone pip package [lenspyx](https://github.com/carronj/lenspyx) (with big speed improvement expected soon)

### Examples

* To obtain analytical reconstruction noise curve and responses, you might want to check [n0s.py](https://github.com/carronj/plancklens/blob/master/plancklens/n0s.py) especially the commented *get_N0* function 

* To use the examples lensing reconstruction parameter files, you will need further to define the environment variable $PLENS to some place safe to write. 
Details on the structure of a parameter file are given in this one: [idealized_example.py](params/idealized_example.py)

* In order to reproduce the 2018 lensing maps and spectrum band-powers, one may use the provided [smicadx12_planck2018.py](params/smicadx12_planck2018.py) parameter file.


* The basics on how to use parameter files can be found in [this jupyter notebook](examples/lensingrec_quickstart.ipynb).
Some details on the numerical computations are collected [in this document](https://arxiv.org/abs/1908.02016).
You might also need to check out the [plancklens documentation](https://plancklens.readthedocs.io/en/latest) (with some bits in progress).

Generally, in plancklens a QE is often described by a short string.

For example 'ptt' stands for lensing (or lensing gradient mode) from temperature x temperature.

Anisotropy source keys are a one-letter string including

    'p' (lensing gradient)

    'x' (lensing curl)

    's' (point sources)

    'f' (modulation field)

    'a' (polarization rotation)

Typical keys include then:

    'ptt', 'xtt', 'stt', 'ftt' for the corresponding QEs from temperature only

    'p_p', 'x_p', 'f_p', 'a_p' for the corresponding QEs from polarization only (combining EE EB and BB if relevant)

    'p', 'x', 'f', 'a', 'f' ... for the MV (or GMV) combination

    'p_eb', ... for the EB estimator (this is the symmetrized version  ('peb' + 'pbe') / 2  so that E and B appear each once on the gradient and inverse-variance filtered leg)

* Jan 14 2021: important bug fix in fortran N1 file affecting EB TB and TE. The n1 module must be rebuilt.

![ERC logo](https://erc.europa.eu/sites/default/files/content/erc_banner-vertical.jpg)
![SNSF logo](./docs/SNF_logo_standard_web_color_neg_e.svg)