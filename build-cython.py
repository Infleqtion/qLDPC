#!/usr/bin/env python3
import os
import shutil

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def build_cython() -> None:
    extension = Extension(
        "*",
        ["*/**/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native", "-Wall"],
    )
    ext_modules = cythonize(extension, compiler_directives={"language_level": "3"})
    distribution = Distribution(dict(ext_modules=ext_modules))

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # copy *.so files to their respective *.pyx directories
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build_cython()
