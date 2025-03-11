import sys

# Flag to check if Fortran extensions should be skipped
skip_fortran = '--no-fortran' in sys.argv

# Remove '--no-fortran' from sys.argv so that the install command runs normally
if skip_fortran:
    sys.argv.remove('--no-fortran')

# Use the correct setup method depending on whether Fortran is included
if skip_fortran:
    import setuptools
    from setuptools import setup

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
        long_description=open("README.md", "r").read(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
else:
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration

    def configuration(parent_package='', top_path=''):
        config = Configuration('', parent_package, top_path)
        config.add_extension('plancklens.wigners.wigners', ['plancklens/wigners/wigners.f90'],
                             extra_link_args=['-lgomp'], libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w'])
        config.add_extension('plancklens.n1.n1f', ['plancklens/n1/n1f.f90'],
                             extra_link_args=['-lgomp'], libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w'])
        return config

    setup(
        name='plancklens',
        version='0.0.1',
        packages=['plancklens', 'plancklens.n1', 'plancklens.filt', 'plancklens.sims', 'plancklens.helpers',
                  'plancklens.qcinv', 'plancklens.wigners', 'tests'],
        data_files=[('plancklens/data/cls', ['plancklens/data/cls/FFP10_wdipole_lensedCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                             'plancklens/data/cls/FFP10_wdipole_params.ini'])],
        url='https://github.com/carronj/plancklens',
        author='Julien Carron',
        author_email='to.jcarron@gmail.com',
        description='Planck lensing python pipeline',
        install_requires=['numpy', 'healpy', 'six', 'scipy'],
        requires=['numpy', 'healpy', 'six', 'scipy'],
        long_description=open("README.md", "r").read(),
        configuration=configuration
    )
