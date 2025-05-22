# Import the Fortran extension
try:
    from . import wigners
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import wigners Fortran extension: {e}. Some functionality may be limited.")