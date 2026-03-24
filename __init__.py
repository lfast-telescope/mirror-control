"""
Mirror control submodule for LFAST telescope.
"""

# Expose main submodules for easier imports
from . import interferometer
from . import SHWFS
from . import shared

__all__ = ['interferometer', 'SHWFS', 'shared']