from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    config.add_extension('plancklens2018.wigners.wigners', ['plancklens2018/wigners/wigners.f90'])
    config.add_extension('plancklens2018.n1.n1f', ['plancklens2018/n1/n1f.f90'],
                         libraries=['gomp'],  extra_compile_args=['-Xpreprocessor', '-fopenmp'])
    return config


setup(
    name='Plancklens2018',
    version='0.0.1',
    packages=['plancklens2018', 'plancklens2018.n1', 'plancklens2018.filt', 'plancklens2018.sims',
              'plancklens2018.qcinv', 'plancklens2018.wigners'],
    url='',
    license='',
    author='Julien Carron',
    author_email='j.carron@sussex.ac.uk',
    description='Planck 2018 lensing python pipeline',
    requires=['numpy', 'healpy'],
    long_description=long_description,
    configuration=configuration)

