# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qLDPC"
author = "Michael A. Perlin"
copyright = "2023 The qLDPC Authors and Infleqtion Inc."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",  # math rendering in html
    "sphinx.ext.napoleon",  # allows google- and numpy- style docstrings
    "IPython.sphinxext.ipython_console_highlighting",
]

# Since our notebooks can involve network I/O (or even costing $), we don't want them to be
# run every time we build the docs. Instead, just use the pre-executed outputs.
nbsphinx_execute = "never"

# In addition, we set the mathjax path to v3, which allows \ket{} (and other commands) to render
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
autosummary_generate = True

templates_path = ["_templates"]

# Using `modules` in index.rst gets the first package and ignores additional included packages.
# Listing out modules explicitly causes building docs to throw error looking for `modules.rst`,
# so add to excluded search patterns as per suggestion here: https://stackoverflow.com/a/15438962
exclude_patterns: list[str] = [
    "modules.rst",
    "setup.rst",
    "general_superstaq.check.rst",
    "cirq_superstaq.ops.rst",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"logo_only": True}
html_css_files = [
    "css/docs-qldpc.css",
]
html_favicon = "_static/logos/Infleqtion_logo.png"
