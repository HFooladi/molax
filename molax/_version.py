"""Version information."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("molax")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "0.0.0.dev0"
except ImportError:
    # Python < 3.8
    __version__ = "0.0.0.dev0"
