# Import the Fortran extension
try:
    # First try to import directly
    from . import n1f
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
        ext_path = os.path.join(dirname, f"n1f{ext_suffix}")

        if os.path.exists(ext_path):
            # Load the extension from the file
            spec = importlib.util.spec_from_file_location("n1f", ext_path)
            n1f = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(n1f)
            globals()["n1f"] = n1f
        else:
            raise ImportError(f"Extension file not found at {ext_path}")
    except ImportError as e2:
        import warnings
        warnings.warn(f"Could not import n1f Fortran extension: {e1}. {e2}. Some functionality may be limited.")