# Import the Fortran extension
try:
    from . import n1f
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import n1f Fortran extension: {e}. Some functionality may be limited.")