# plancklens

plancklens is is a python code for cosmology containing most of Planck 2018 CMB lensing pipeline, by Julien Carron. See the publication https://arxiv.org/abs/1807.06210.
Some numerical parts are written in Fortran. Portions of it (structure and code) have been directly adapted from pre-existing work by Duncan Hanson.

*Installation:*
 
After cloning the repository, build an editable installation with
    
    pip install -e . [--user]

The –-user is required only if you don’t have write permission to your main python installation. A fortran compiler is required for a successful installation.

To use the examples lensing reconstruction parameter files, you will need further to define the environment variable $PLENS to some place safe to write.

To reproduce the 2018 lensing spectrum band-powers, use the parameter file

    params/smicadx12_planck2018.py
    
Details on the structure of a parameter file are given in
    
    params/idealized_example.py

Some details on the numerical computations are collected `here <https://github.com/carronj/plancklens/blob/master/supplement.pdf>`_
![alt text](https://erc.europa.eu/sites/default/files/content/erc_banner-vertical.jpg)