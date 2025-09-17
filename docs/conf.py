# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

import geobench_v2

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_togglebutton",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"

# List of source suffix to include .py for jupytext
source_suffix = [".rst", ".md", ".py"]
# Alternative format:
# source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]

source_dirs = ["api"]

# General information about the project.
project = "GEO-Bench-2"
copyright = "2025, The AI Alliance"
version = geobench_v2.__version__
release = geobench_v2.__version__

# Exclude ipynb for jupytext
exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "GEO-Bench-2"

# Uncomment to use logo
# html_logo = "_static/logo.jpeg"
# html_favicon = "_static/logo.jpeg"

html_static_path = ["_static"]
# Uncomment to use custom CSS
html_css_files = ["my_theme.css"]

html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/The-AI-Alliance/GEO-Bench-2",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "navigation_with_keys": True,
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Do not execute notebooks during the build
nbsphinx_execute = "never"
