"""LemonMatrix: a benchmarking control plane for Lemonade instances."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lemonmatrix")
except PackageNotFoundError:
    __version__ = "0+unknown"
