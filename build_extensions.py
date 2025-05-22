#!/usr/bin/env python
import os
import subprocess
import sys
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def build_extension(source_file, output_dir):
    """Build a Fortran extension using f2py."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the module name from the source file
    module_name = os.path.splitext(os.path.basename(source_file))[0]

    # Try to use meson if available, otherwise fall back to direct f2py
    if is_package_installed('meson') and is_package_installed('ninja'):
        print(f"Using meson backend for {module_name}")
        try:
            # Use f2py with meson backend
            cmd = [
                'f2py',
                '-c',
                source_file,
                '-m', module_name,
                '--f90flags=-fopenmp -w -fPIC',
                '-lgomp'
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)

            # Move the extension to the output directory
            extension_file = f"{module_name}{get_extension_suffix()}"
            if os.path.exists(extension_file):
                dest_file = os.path.join(output_dir, extension_file)
                print(f"Moving {extension_file} to {dest_file}")
                os.rename(extension_file, dest_file)
                return True
            else:
                print(f"Warning: Extension file {extension_file} not found")
                return False
        except Exception as e:
            print(f"Warning: Failed to build extension {module_name} with meson: {e}")
            print("Falling back to direct f2py...")
    else:
        print(f"Meson or ninja not found, using direct f2py for {module_name}")

    # Direct f2py approach
    try:
        # Build the extension directly with f2py
        cmd = [
            'f2py',
            '-c',
            source_file,
            '-m', module_name,
            '--f90flags=-fopenmp -w -fPIC',
            '-lgomp'
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        # Move the extension to the output directory
        extension_file = f"{module_name}{get_extension_suffix()}"
        if os.path.exists(extension_file):
            dest_file = os.path.join(output_dir, extension_file)
            print(f"Moving {extension_file} to {dest_file}")
            os.rename(extension_file, dest_file)
            return True
        else:
            print(f"Warning: Extension file {extension_file} not found")
            return False
    except Exception as e:
        print(f"Error: Failed to build extension {module_name}: {e}")
        return False

def get_extension_suffix():
    """Get the extension suffix for the current Python version."""
    import sysconfig
    return sysconfig.get_config_var('EXT_SUFFIX')

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to the script directory
    original_dir = os.getcwd()
    os.chdir(script_dir)

    success = True
    try:
        # Build the wigners extension
        if not build_extension('plancklens/wigners/wigners.f90', 'plancklens/wigners'):
            success = False

        # Build the n1f extension
        if not build_extension('plancklens/n1/n1f.f90', 'plancklens/n1'):
            success = False
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
