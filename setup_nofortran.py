import setuptools
from setuptools import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='plancklens',
    version='0.0.1',
    packages=setuptools.find_packages(),
    data_files=[('plancklens/data/cls', ['plancklens/data/cls/FFP10_wdipole_lensedCls.dat',
                                         'plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                         'plancklens/data/cls/FFP10_wdipole_params.ini'])],
    url='https://github.com/carronj/plancklens',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='Planck lensing python pipeline',
    install_requires=['numpy', 'healpy', 'six', 'scipy'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
