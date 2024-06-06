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
copyright = "2023 The qLDPC Authors and Infleqtion Inc."
author = "Michael A. Perlin"

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

# Our notebooks can involve network I/O (or even costing $), so we don't want them to be
# run every time we build the docs.  Instead, just use the pre-executed outputs.
nbsphinx_execute = "never"

# In addition, we set the mathjax path to v3, which allows \ket{} (and other commands) to render
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"logo_only": True}
html_favicon = "_static/logos/Infleqtion_logo.png"
