import pkg_resources

from . import abstract, codes, decoder, objects

decode = decoder.decode

__version__ = pkg_resources.get_distribution("qldpc").version

__all__ = [
    "__version__",
    "abstract",
    "circuits",
    "codes",
    "decode",
    "decoder",
    "objects",
]
