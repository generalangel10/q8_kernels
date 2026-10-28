"""Public package metadata and lazy access to CUDA-backed subpackages."""

from importlib import import_module

from ._version import __version__

__all__ = ["functional", "models", "__version__"]


def __getattr__(name):
    """Load CUDA-backed subpackages only when they are explicitly requested.

    Keeping package metadata importable on CPU-only machines makes repository
    tooling, version checks, and documentation builds possible without
    pretending that the CUDA extensions themselves are available.
    """
    if name in {"functional", "models"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
