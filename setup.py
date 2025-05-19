import sys
import os

# Flag to check if Fortran extensions should be skipped
skip_fortran = '--no-fortran' in sys.argv

# Remove '--no-fortran' from sys.argv so that the install command runs normally
if skip_fortran:
    sys.argv.remove('--no-fortran')

# Use the correct setup method depending on whether Fortran is included
if skip_fortran:
    import setuptools
    from setuptools import setup

    # Read the long description from README.md
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name='plancklens',
        version='0.1.0',  # Updated version
        packages=setuptools.find_packages(),
        data_files=[('plancklens/data/cls', ['plancklens/data/cls/FFP10_wdipole_lensedCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_params.ini'])],
        url='https://github.com/carronj/plancklens',
        author='Julien Carron',
        author_email='to.jcarron@gmail.com',
        description='Planck lensing python pipeline',
        install_requires=['numpy>=1.20.0', 'healpy>=1.15.0', 'six', 'scipy>=1.7.0'],
        python_requires='>=3.8',
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
else:
    # For NumPy 2.0+, we need to use setuptools with numpy.distutils
    import setuptools
    from setuptools import setup
    from numpy.distutils.core import Extension
    from numpy.distutils.misc_util import Configuration

    # Read the long description from README.md
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    def configuration(parent_package='', top_path=''):
        config = Configuration('', parent_package, top_path)
        config.add_extension('plancklens.wigners.wigners', ['plancklens/wigners/wigners.f90'],
                             extra_link_args=['-lgomp'], libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w'])
        config.add_extension('plancklens.n1.n1f', ['plancklens/n1/n1f.f90'],
                             extra_link_args=['-lgomp'], libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w'])
        return config

    setup(
        name='plancklens',
        version='0.1.0',  # Updated version
        packages=['plancklens', 'plancklens.n1', 'plancklens.filt', 'plancklens.sims', 'plancklens.helpers',
                  'plancklens.qcinv', 'plancklens.wigners', 'tests'],
        data_files=[('plancklens/data/cls', ['plancklens/data/cls/FFP10_wdipole_lensedCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_params.ini'])],
        url='https://github.com/carronj/plancklens',
        author='Julien Carron',
        author_email='to.jcarron@gmail.com',
        description='Planck lensing python pipeline',
        install_requires=['numpy>=1.20.0', 'healpy>=1.15.0', 'six', 'scipy>=1.7.0'],
        python_requires='>=3.8',
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        configuration=configuration
    )
