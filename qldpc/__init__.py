import importlib.metadata

from . import abstract, codes, decoder, external, objects, simulator

__version__ = importlib.metadata.version("qldpc")

__all__ = [
    "__version__",
    "abstract",
    "codes",
    "decoder",
    "external",
    "objects",
    "simulator",
]
