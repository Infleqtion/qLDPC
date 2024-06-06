# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../qldpc"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qLDPC"
copyright = "2023 The qLDPC Authors and Infleqtion Inc."  # pylint:disable=redefined-builtin
author = "Michael A. Perlin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Our notebooks can involve network I/O (or even costing $), so we don't want them to be
# run every time we build the docs.  Instead, just use the pre-executed outputs.
nbsphinx_execute = "never"

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["modules.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
