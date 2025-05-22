import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Flag to check if Fortran extensions should be skipped
skip_fortran = '--no-fortran' in sys.argv

# Remove '--no-fortran' from sys.argv so that the install command runs normally
if skip_fortran:
    sys.argv.remove('--no-fortran')

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Custom build_ext command for Fortran extensions
class CustomBuildExt(build_ext):
    def run(self):
        # Run the build_extensions method
        self.build_extensions()

    def build_extension(self, ext):
        if ext.name in ['plancklens.wigners.wigners', 'plancklens.n1.n1f']:
            # Check if the extension is already built by the build_extensions.py script
            ext_path = self.get_ext_fullpath(ext.name)
            ext_dir = os.path.dirname(ext_path)
            module_name = os.path.splitext(os.path.basename(ext.sources[0]))[0]

            # Look for the extension with the platform-specific suffix
            import importlib.machinery
            ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            built_ext = os.path.join(os.path.dirname(ext.sources[0]), f"{module_name}{ext_suffix}")

            if os.path.exists(built_ext):
                # If the extension is already built, copy it to the build directory
                os.makedirs(ext_dir, exist_ok=True)
                import shutil
                shutil.copy2(built_ext, ext_path)
                print(f"Copied existing extension {built_ext} to {ext_path}")
            else:
                # If the extension is not built, use the build_extensions.py script
                import subprocess
                subprocess.check_call([sys.executable, 'build_extensions.py'])

                # Check if the extension was built successfully
                if os.path.exists(built_ext):
                    # Copy the extension to the build directory
                    os.makedirs(ext_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(built_ext, ext_path)
                    print(f"Copied built extension {built_ext} to {ext_path}")
                else:
                    # If the extension was not built, raise an error
                    raise RuntimeError(f"Failed to build extension {ext.name}")
        else:
            # Use the default build_extension for other extensions
            super().build_extension(ext)

# Define the extensions
extensions = []
if not skip_fortran:
    extensions = [
        Extension('plancklens.wigners.wigners', ['plancklens/wigners/wigners.f90']),
        Extension('plancklens.n1.n1f', ['plancklens/n1/n1f.f90']),
    ]

# Setup configuration
setup(
    name='plancklens',
    version='0.1.0',
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
    ext_modules=extensions if not skip_fortran else [],
    cmdclass={'build_ext': CustomBuildExt} if not skip_fortran else {},
)
