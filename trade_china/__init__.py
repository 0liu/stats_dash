import sys

if sys.version_info < (3, 7, 1):
    raise RuntimeError("Python 3.7.1 or higher")

__version__ = '2.0.0'

from .dashboard import Dashboard
__all__ = ['Dashboard',]
