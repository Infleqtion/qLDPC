import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension

import numpy
from Cython.Build import cythonize

compile_args = []
link_args = []
include_dirs = [numpy.get_include()]
libraries = []


def build():
    extensions = [
        Extension(
            "*",
            ["*/**/*.pyx"],
            language="c",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            libraries=libraries,
        )
    ]
    ext_modules = cythonize(
        extensions,
        include_path=include_dirs,
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = "extended"

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
