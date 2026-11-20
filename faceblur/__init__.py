"""Mask one chosen face throughout a video."""

try:  # written into published zips by install/publish.py; absent in a git checkout
    from ._version import __version__
except ImportError:
    __version__ = "dev"
