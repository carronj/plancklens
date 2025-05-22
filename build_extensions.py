#!/usr/bin/env python
import os
import subprocess
import sys

def build_extension(source_file, output_dir):
    """Build a Fortran extension using f2py."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the module name from the source file
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    
    # Build the extension
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
    else:
        print(f"Warning: Extension file {extension_file} not found")

def get_extension_suffix():
    """Get the extension suffix for the current Python version."""
    import sysconfig
    return sysconfig.get_config_var('EXT_SUFFIX')

def main():
    # Build the wigners extension
    build_extension('plancklens/wigners/wigners.f90', 'plancklens/wigners')
    
    # Build the n1f extension
    build_extension('plancklens/n1/n1f.f90', 'plancklens/n1')

if __name__ == '__main__':
    main()
