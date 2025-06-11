# Import the Fortran extension
try:
    # First try to import directly
    from . import wigners
except ImportError as e1:
    try:
        # Try to find the extension with the platform-specific suffix
        import os
        import importlib.util
        import importlib.machinery

        # Get the directory of this file
        dirname = os.path.dirname(__file__)

        # Get the extension suffix for the current Python version
        ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

        # Look for the extension file
        ext_path = os.path.join(dirname, f"wigners{ext_suffix}")

        if os.path.exists(ext_path):
            # Load the extension from the file
            spec = importlib.util.spec_from_file_location("wigners", ext_path)
            wigners = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wigners)
            globals()["wigners"] = wigners
        else:
            raise ImportError(f"Extension file not found at {ext_path}")
    except ImportError as e2:
        import warnings
        warnings.warn(f"Could not import wigners Fortran extension: {e1}. {e2}. Some functionality may be limited.")