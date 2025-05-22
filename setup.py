import os
import sys
import importlib.machinery
import importlib.util
import subprocess
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
        # Build the Fortran extensions first
        if not skip_fortran:
            try:
                # Ensure meson and ninja are installed
                try:
                    import meson
                except ImportError:
                    print("Installing meson and ninja...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'meson', 'ninja'])
                    print("Meson and ninja installed successfully.")

                print("Building Fortran extensions using build_extensions.py...")
                subprocess.check_call([sys.executable, 'build_extensions.py'])
                print("Fortran extensions built successfully.")
            except Exception as e:
                print(f"Warning: Failed to build Fortran extensions: {e}")
                print("Continuing without Fortran extensions...")

        # Run the build_extensions method
        self.build_extensions()

    def build_extension(self, ext):
        if ext.name in ['plancklens.wigners.wigners', 'plancklens.n1.n1f']:
            # Get the extension path
            ext_path = self.get_ext_fullpath(ext.name)
            ext_dir = os.path.dirname(ext_path)
            module_name = os.path.splitext(os.path.basename(ext.sources[0]))[0]

            # Look for the extension with the platform-specific suffix
            ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            built_ext = os.path.join(os.path.dirname(ext.sources[0]), f"{module_name}{ext_suffix}")

            if os.path.exists(built_ext):
                # If the extension is already built, copy it to the build directory
                os.makedirs(ext_dir, exist_ok=True)
                import shutil
                shutil.copy2(built_ext, ext_path)
                print(f"Copied extension {built_ext} to {ext_path}")
            else:
                # If the extension was not built, raise a warning
                print(f"Warning: Extension {ext.name} not found at {built_ext}")
                print("Continuing without this extension...")
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
    package_data={
        'plancklens.wigners': ['*.so'],
        'plancklens.n1': ['*.so'],
        'plancklens.data.cls': ['*.dat', '*.ini'],
    },
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
    include_package_data=True,
)
