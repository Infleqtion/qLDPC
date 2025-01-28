import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="test",
        sources=["test.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++11"],
    )
]

setup(ext_modules=cythonize(extensions))
