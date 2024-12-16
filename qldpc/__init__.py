import importlib.metadata

from . import abstract, cache, circuits, codes, decoder, external, objects

__version__ = importlib.metadata.version("qldpc")

__all__ = [
    "__version__",
    "abstract",
    "cache",
    "circuits",
    "codes",
    "decoder",
    "external",
    "objects",
]
