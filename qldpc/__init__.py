import importlib.metadata

from . import abstract, codes, decoder, objects

decode = decoder.decode

__version__ = importlib.metadata.version("qldpc")

__all__ = [
    "__version__",
    "abstract",
    "circuits",
    "codes",
    "decode",
    "decoder",
    "objects",
]
