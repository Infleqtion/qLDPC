#!/usr/bin/env python3
import distutils.ccompiler
import glob
import os
import shutil
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def build_cython(*extra_compile_args: str, rebuild: bool = False) -> None:
    pyx_files = glob.glob(os.path.join(os.path.dirname(__file__), "**", "*.pyx"), recursive=True)
    extension = Extension(
        "*",
        pyx_files,
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=list(extra_compile_args),
    )
    ext_modules = cythonize(extension, compiler_directives={"language_level": "3"}, force=rebuild)
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
    # identify extra arguments to pass to the C compiler
    extra_compile_args = sys.argv[1:]

    # determine whether to force-rebuild cython code
    rebuild = "--rebuild" in extra_compile_args
    if rebuild:
        extra_compile_args.remove("--rebuild")

    # if no compiler arguments are provided, set some defaults
    if not extra_compile_args:
        if distutils.ccompiler.get_default_compiler() == "msvc":
            extra_compile_args == ["/O2"]
        else:
            extra_compile_args = ["-O3", "-march=native", "-Wall"]

    build_cython(*extra_compile_args, rebuild=rebuild)
