#!/usr/bin/env python3
import distutils.ccompiler
import os
import shutil

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

if distutils.ccompiler.get_default_compiler() == "msvc":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3", "-march=native", "-Wall"]


def build_cython() -> None:
    extension = Extension(
        "*",
        ["*/**/*.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
    )
    ext_modules = cythonize(extension, compiler_directives={"language_level": "3"})
    distribution = Distribution(dict(ext_modules=ext_modules))

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # copy *.so files to their respective *.pyx directories
    for output in cmd.get_outputs():
        destination = os.path.relpath(output, cmd.build_lib)
        if os.path.isfile(destination):
            os.remove(destination)
        shutil.copyfile(output, destination)


if __name__ == "__main__":
    build_cython()
